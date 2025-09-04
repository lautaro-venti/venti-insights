# -*- coding: utf-8 -*-
"""
Venti – Insight IA: Reporte semanal Intercom (Notion narrativo + Gemini API)
- KPIs debajo de Resumen Ejecutivo (tarjetas + bullets)
- IA con modos (full/lite/off), presupuesto y cache para evitar 429
- Sanitizado de links y tablas nativas Notion
"""

import os
import re
import json
import argparse
import subprocess
import shutil
import hashlib
from datetime import date
import requests
import pandas as pd
import numpy as np
import unicodedata
import time

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== Utilidades base =====================

EMOJI_RX = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)

def _safe_str(v: object) -> str:
    try:
        if v is None: 
            return ""
        # pandas NaN
        if isinstance(v, float) and pd.isna(v):
            return ""
        s = str(v)
        if s.lower() == "nan":
            return ""
        return s
    except Exception:
        return ""

def strip_emojis(s: str) -> str:
    try:
        return EMOJI_RX.sub("", s or "")
    except Exception:
        return s or ""

def _norm_txt(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s

def load_csv_robusto(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1", "utf-16", "utf-16-le", "utf-16-be"]
    seps = [",", ";", "\t", None]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python", dtype=str, on_bad_lines="skip")
                else:
                    df = pd.read_csv(csv_path, encoding=enc, sep=sep, engine="python", dtype=str, on_bad_lines="skip")
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last_err = e
                continue
    try:
        return pd.read_excel(csv_path, dtype=str)
    except Exception:
        pass
    raise RuntimeError(f"No pude leer el CSV. Último error: {last_err}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        cc = str(c).strip().lower()
        cc = cc.replace(" ", "_").replace("__", "_")
        cc = (cc
              .replace("í", "i").replace("á", "a").replace("é", "e")
              .replace("ó", "o").replace("ú", "u").replace("ñ", "n"))
        new_cols.append(cc)
    df.columns = new_cols

    aliases = {
        "insight_ia": ["insight", "insightia", "insight_ia"],
        "resumen_ia": ["resumen", "resumenia", "resumen_ia"],
        "palabras_clave": ["palabras_clave","palabrasclave","keywords"],
        "canal": ["canal","channel"],
        "area": ["area","área","area_","area__"],
        "tema": ["tema","topic"],
        "motivo": ["motivo","reason"],
        "submotivo": ["submotivo","sub_reason","submot"],
        "urgencia": ["urgencia","priority","severity"],
        "sentimiento": ["sentimiento","sentiment"],
        "categoria": ["categoria","categoria_","categoría","categoria__"],
        "link_a_intercom": ["link_a_intercom","link_intercom","link","url_intercom","link__a__intercom"],
        "id_intercom": ["id_intercom","id","conversation_id"],
        "fecha": ["fecha","date","created_at"],
        "rol": ["rol","role"],
        # KPI crudos / timing
        "created_at": ["created_at","created","createdat"],
        "first_admin_reply_at": ["first_admin_reply_at","firstadminreplyat","first_admin_reply","first_response_at"],
        "first_contact_reply_at": ["first_contact_reply_at","firstcontactreplyat","statistics_first_contact_reply_at","first_contact_reply_created_at"],
        "closed_at": ["closed_at","closed","first_close_at","statistics_first_close_at","statistics_last_close_at"],
        "ttr_seconds": ["ttr_seconds","time_to_first_close","statistics_time_to_first_close"],
        "first_response_seconds": ["first_response_seconds","first_response_time","first_reply_seconds"],
        "status": ["status","state"],
        "csat": ["csat","rating","conversation_rating","rating_value"],
    }
    present = set(df.columns)
    for canon, options in aliases.items():
        if canon in present:
            continue
        for opt in options:
            if opt in present:
                df.rename(columns={opt: canon}, inplace=True)
                present.add(canon)
                break

    base_cols = ["resumen_ia","insight_ia","palabras_clave","canal","area","tema",
                 "motivo","submotivo","urgencia","sentimiento","categoria",
                 "link_a_intercom","id_intercom","fecha","rol"]
    for canon in base_cols:
        if canon not in df.columns:
            df[canon] = ""
    return df

# ===================== Taxonomías =====================

VALID_TEMAS = {
    "eventos - user ticket","eventos - user productora","lead comercial","anuncios & notificaciones",
    "duplicado","desvío a intercom","sin respuesta",
}
VALID_MOTIVOS = {
    "caso excepcional","reenvío","estafa por reventa","compra externa a venti","consulta por evento",
    "team leads & públicas","devolución","pagos","seguridad","evento reprogramado","evento cancelado",
    "contacto comercial","anuncios & notificaciones","duplicado","desvío a intercom","no recibí mi entrada",
    "sdu (sist. de usuarios)","transferencia de entradas","qr shield","venti swap","reporte","carga masiva",
    "envío de invitaciones","carga de un evento","servicios operativos","solicitud de reembolso","adelantos",
    "liquidaciones","estado de cuenta","datos de cuenta","altas en venti","app de validación","validadores",
    "organización de accesos en el evento","facturación","sin respuesta","reclamo de usuario",
    "consulta sobre uso de la plataforma","desvinculación de personal",
}
VALID_SUBMOTIVOS = set()

def map_to_catalog(value, catalog):
    v = _norm_txt(value)
    if not v or v in ("nan","none"):
        return "", False
    norm_catalog = {_norm_txt(x): x for x in catalog}
    if v in norm_catalog:
        return norm_catalog[v], True
    for raw in catalog:
        nraw = _norm_txt(raw)
        if v in nraw or nraw in v:
            return raw, True
    return value if isinstance(value, str) else str(value), False

def enforce_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    df["tema_norm"], df["tema_ok"] = zip(*df["tema"].map(lambda x: map_to_catalog(x, VALID_TEMAS)))
    df["motivo_norm"], df["motivo_ok"] = zip(*df["motivo"].map(lambda x: map_to_catalog(x, VALID_MOTIVOS)))
    if "submotivo" in df.columns and len(VALID_SUBMOTIVOS) > 0:
        df["submotivo_norm"], df["submotivo_ok"] = zip(*df["submotivo"].map(lambda x: map_to_catalog(x, VALID_SUBMOTIVOS)))
        df["taxonomy_flag"] = ~(df["tema_ok"] & df["motivo_ok"] & df["submotivo_ok"])
    else:
        df["submotivo_norm"] = df["submotivo"]
        df["taxonomy_flag"] = ~(df["tema_ok"] & df["motivo_ok"])
    return df

def build_text_base(row: pd.Series) -> str:
    parts = []
    for col in ["resumen_ia","insight_ia","palabras_clave","tema_norm","motivo_norm","submotivo_norm","area"]:
        val = _norm_txt(row.get(col,""))
        if val and val != "nan":
            parts.append(val)
    return " | ".join(parts)

# ----------------- NUEVO: CSAT, ESTADO FINAL e ISSUE BUCKETS -----------------

def pick_csats(row):
    for k in ["csat", "csat_ic", "csat_intercom", "csat_ia", "csat_modelo", "csat_gemini"]:
        if k in row and pd.notna(row[k]):
            try:
                v = int(float(row[k]))
                if 1 <= v <= 5:
                    return v
            except:
                pass
    return np.nan

def estado_final_from(row):
    if "estado_final" in row and pd.notna(row["estado_final"]):
        return str(row["estado_final"]).strip().capitalize()

    status = str(row.get("status","")).strip().lower()
    resumen = str(row.get("resumen_ia","") or row.get("resumen","")).strip().lower()

    if "estado final: resuelto" in resumen or "resuelto" in resumen:
        return "Resuelto"
    if "estado final: pendiente" in resumen or "pendiente" in resumen:
        return "Pendiente"
    if "sin respuesta" in resumen or "estado final: sin respuesta" in resumen:
        return "Sin respuesta"

    if status == "closed":
        return "Resuelto"
    if status in {"open","snoozed"}:
        return "Pendiente"
    return "No resuelto"

ISSUE_MAP = {
    "No recibí mi entrada": "Entrega de entradas",
    "QR Shield": "QR / Validación en acceso",
    "Pagos": "Pagos / cobros",
    "Venti Swap": "Venti Swap",
    "Servicios operativos": "App / rendimiento / bug",
    "Consulta por evento": "Información de evento",
    "Transferencia de entradas": "Transferencia / titularidad",
    "Devolución": "Reembolso / devolución",
    "Contacto Comercial": "Comercial",
    "_default": "Otros",
}
ISSUE_REGEX_RULES = [
    ("Entrega de entradas", r"(no\s*recib|reenv[ií]o|link\s*de\s*entrada|entrada(s)?\s*(no)?\s*llega|ticket\s*no|no\s*me\s*ll[eé]g[oó])"),
    ("Transferencia / titularidad", r"(transferenc|cambio\s*de\s*titular|modificar\s*(nombre|titular)|pasar\s*entrada)"),
    ("QR / Validación en acceso", r"\bqr\b|validaci[oó]n|validad(or|ores)|escane"),
    ("Pagos / cobros", r"\bpago(s)?\b|cobro|rechazad|tarjeta|mercadopago|\bmp\b|cuotas"),
    ("Reembolso / devolución", r"reembols|devoluci[oó]n|refund|chargeback"),
    ("Cuenta / login / registro", r"cuenta|login|logue|registr|contrase[nñ]a|clave|verificaci[oó]n\s*de\s*mail|correo\s*inv[aá]lido"),
    ("App / rendimiento / bug", r"\bapp\b|aplicaci[oó]n|crash|no\s*funciona|bug|error\s*(t[eé]cnico|500|404)"),
    ("Soporte / sin respuesta / SDU", r"sin\s*respuesta|\bsdu\b|jotform|demora|espera"),
    ("Información de evento", r"consulta\s*por\s*evento|horario|ubicaci[oó]n|vip|mapa|line\s*up|ingreso|puerta"),
    ("Productores / RRPP / invitaciones", r"invitaci[oó]n|\brrpp\b|productor|productora|validadores|operativo"),
]

def choose_issue(motivo_final, submotivo_final, texto_base):
    m = _safe_str(motivo_final).strip()
    if m in ISSUE_MAP:
        return ISSUE_MAP[m]

    sm = _safe_str(submotivo_final).strip().lower()
    if "qr" in sm:
        return "QR / Validación en acceso"
    if "pago" in sm or "mp" in sm:
        return "Pagos / cobros"

    t = _norm_txt(_safe_str(texto_base))
    for label, rx in ISSUE_REGEX_RULES:
        if re.search(rx, t):
            return label
    return ISSUE_MAP["_default"]

# ===================== Heurísticas de Issues =====================

RULES = [
    ("Entrega de entradas", r"(no\s*recibi|reenvio|link\s*de\s*entrada|entrada(s)?\s*(no)?\s*llega|ticket\s*no\s|no\s*me\s*llego)"),
    ("Transferencia / titularidad", r"(transferenc|transferir|cambio\s*de\s*titular|modificar\s*(nombre|titular)|pasar\s*entrada)"),
    ("QR / Validación en acceso", r"(qr|validacion|control\s*de\s*acceso|escaneo|lector|validador)"),
    ("Pagos / cobros", r"(pago|pagos|cobro|cobrar|rechazad|tarjeta|mercadopago|\bmp\b|cuotas)"),
    ("Reembolso / devolución", r"(reembolso|devolucion|refund|chargeback)"),
    ("Cuenta / login / registro", r"(cuenta|logue|login|registr|contrasena|clave|verificacion\s*de\s*mail|correo\s*invalido)"),
    ("App / rendimiento / bug", r"(app|aplicacion|crash|se\s*cierra|no\s*funciona|bug|error\s*(tecnico|500|404))"),
    ("Soporte / sin respuesta / SDU", r"(sin\s*respuesta|derivacion\s*al\s*sdu|\bsdu\b|jotform|demora|espera)"),
    ("Información de evento", r"(consulta\s*por\s*evento|horario|ubicacion|vip|mapa|line\s*up|capacidad|ingreso|puerta)"),
    ("Productores / RRPP / invitaciones", r"(invitacion|\brrpp\b|productor|productora|acceso\s*productor|carga\s*de\s*evento|validadores|operativo)"),
]

def assign_issue_group(text: str) -> str:
    t = _norm_txt(text)
    for label, pattern in RULES:
        if re.search(pattern, t):
            return label
    return "Otros"

def top_values(series: pd.Series, n=3) -> str:
    vc = series.fillna("").replace("nan","").astype(str).str.strip().value_counts()
    items = [f"{idx} ({cnt})" for idx, cnt in vc.head(n).items() if idx]
    return ", ".join(items)

def _safe_lower(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return str(x).strip().lower()
    except Exception:
        return ""

def compute_flags(row: pd.Series) -> pd.Series:
    txt = " ".join(str(row.get(c,"") or "") for c in
                   ["resumen_ia","insight_ia","palabras_clave","tema_norm","motivo_norm","submotivo_norm"]).lower()
    urg = _safe_lower(row.get("urgencia",""))
    canal = _safe_lower(row.get("canal",""))
    risk = "LOW"
    if any(k in _norm_txt(txt) for k in ["estafa","fraude","no puedo entrar","no puedo ingresar","rechazad"]):
        risk = "HIGH"
    elif urg in ("alta","high"):
        risk = "HIGH"
    elif urg in ("media","medium"):
        risk = "MEDIUM"
    sla_now = (canal in ("whatsapp","instagram")) and (risk == "HIGH")
    return pd.Series({"risk": risk, "sla_now": sla_now})

# ===================== KPI desde campos crudos =====================

def _to_num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan]*len(df))

def compute_kpis_from_raw_df(df: pd.DataFrame, sla_minutes: int = 15) -> dict:
    created  = _to_num(df, "created_at")
    closed   = _to_num(df, "closed_at")
    far      = _to_num(df, "first_admin_reply_at")
    fcr      = _to_num(df, "first_contact_reply_at")
    frs      = _to_num(df, "first_response_seconds")
    ttr_secs = _to_num(df, "ttr_seconds")

    # first response seconds
    need_frs = frs.isna()
    base_first = far.where(~far.isna(), fcr)
    est_frs = base_first - created
    frs = frs.where(~need_frs, est_frs)
    frs = frs.where(frs >= 0, np.nan)

    # TTR
    need_ttr = ttr_secs.isna()
    est_ttr = closed - created
    ttr_secs = ttr_secs.where(~need_ttr, est_ttr)
    ttr_secs = ttr_secs.where(ttr_secs >= 0, np.nan)

    # Tickets resueltos (fallback por status, se refuerza luego con estado_final)
    if "status" in df.columns:
        tickets = int((df["status"].astype(str).str.lower() == "closed").sum()) or int(len(df))
    else:
        tickets = int(len(df))

    # % primera respuesta
    base = int(frs.notna().sum())
    ok = int((frs <= sla_minutes*60).sum()) if base else 0
    tasa_1ra = (ok/base*100.0) if base else None
    fr_p50 = float(np.nanmedian(frs)) if base else None

    # TTR (h)
    if ttr_secs.notna().any():
        ttr_mean_h = float(np.nanmean(ttr_secs))/3600.0
        ttr_p50_h  = float(np.nanpercentile(ttr_secs.dropna(), 50))/3600.0
        ttr_p90_h  = float(np.nanpercentile(ttr_secs.dropna(), 90))/3600.0
    else:
        ttr_mean_h = ttr_p50_h = ttr_p90_h = None

    # CSAT
    csat = None; csat_n = 0
    if "csat" in df.columns:
        cs = pd.to_numeric(df["csat"], errors="coerce")
        csat_n = int(cs.notna().sum())
        if cs.notna().any():
            csat = round(float(cs.mean()), 2)

    return {
        "tickets_resueltos": tickets,
        "tasa_1ra_respuesta": float(tasa_1ra) if isinstance(tasa_1ra, (int,float)) else None,
        "first_base": base,
        "first_ok": ok,
        "first_resp_p50_s": float(fr_p50) if isinstance(fr_p50, float) else None,
        "ttr_horas": float(ttr_mean_h) if isinstance(ttr_mean_h, float) else None,
        "ttr_p50_h": float(ttr_p50_h) if isinstance(ttr_p50_h, float) else None,
        "ttr_p90_h": float(ttr_p90_h) if isinstance(ttr_p90_h, float) else None,
        "csat": csat,
        "csat_n": csat_n,
        "sla_minutes": sla_minutes,
    }

def _fmt_min_from_sec(s):
    if s is None: return "—"
    return f"{int(round(s/60.0))} min"

def _fmt_h(h):
    if h is None: return "—"
    return f"{h:.2f} h"

# ===================== Gráficos =====================

def chart_top_issues(df, out_dir):
    counts = df["issue_group"].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(8,5))
    labels = list(counts.index)
    vals = list(counts.values)
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_title("Top Issues"); ax.set_xlabel("Casos")
    ax.invert_yaxis()
    p = os.path.join(out_dir, "top_issues.png")
    plt.tight_layout(); fig.savefig(p); plt.close(fig)
    return p, counts

def chart_urgencia_pie(df, out_dir):
    counts = df["urgencia"].fillna("Sin dato").replace({"nan":"Sin dato"}).value_counts()
    labels = list(counts.index); vals = list(counts.values)
    fig, ax = plt.subplots(figsize=(6,6))
    if len(vals) == 0:
        vals = [1]; labels = ["Sin datos"]
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Distribución de Urgencias")
    p = os.path.join(out_dir, "urgencia_pie.png")
    plt.tight_layout(); fig.savefig(p); plt.close(fig)
    return p, counts

def chart_sentimiento_pie(df, out_dir):
    counts = df["sentimiento"].fillna("Sin dato").replace({"nan":"Sin dato"}).value_counts()
    labels = list(counts.index); vals = list(counts.values)
    fig, ax = plt.subplots(figsize=(6,6))
    if len(vals) == 0:
        vals = [1]; labels = ["Sin datos"]
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Distribución de Sentimientos")
    p = os.path.join(out_dir, "sentimiento_pie.png")
    plt.tight_layout(); fig.savefig(p); plt.close(fig)
    return p, counts

def chart_urgencia_por_issue(df, out_dir):
    ct = pd.crosstab(df["issue_group"], df["urgencia"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(9,6))
    x = np.arange(len(ct.index))
    bottom = np.zeros(len(ct.index))
    for col in ct.columns:
        vals = ct[col].values
        ax.bar(x, vals, bottom=bottom, label=str(col))
        bottom = bottom + vals
    ax.set_xticks(x); ax.set_xticklabels(list(ct.index), rotation=45, ha="right")
    ax.set_title("Urgencia por Issue"); ax.set_ylabel("Casos"); ax.legend()
    p = os.path.join(out_dir, "urgencia_por_issue.png")
    plt.tight_layout(); fig.savefig(p); plt.close(fig)
    return p, ct

def chart_canal_por_issue(df, out_dir):
    ct = pd.crosstab(df["issue_group"], df["canal"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(9,6))
    x = np.arange(len(ct.index))
    bottom = np.zeros(len(ct.index))
    for col in ct.columns:
        vals = ct[col].values
        ax.bar(x, vals, bottom=bottom, label=str(col))
        bottom = bottom + vals
    ax.set_xticks(x); ax.set_xticklabels(list(ct.index), rotation=45, ha="right")
    ax.set_title("Canal por Issue"); ax.set_ylabel("Casos"); ax.legend(ncols=2, fontsize=8)
    p = os.path.join(out_dir, "canal_por_issue.png")
    plt.tight_layout(); fig.savefig(p); plt.close(fig)
    return p, ct

def chart_urgencias_en_top_issues(df, out_dir, top_n=5):
    top = df["issue_group"].value_counts().head(top_n).index.tolist()
    sub = df[df["issue_group"].isin(top)].copy()
    urg_levels = ["Alta","Media","Baja"]
    sub["urgencia_norm"] = sub["urgencia"].fillna("Sin dato").replace({"nan":"Sin dato"}).str.title()
    sub["urgencia_norm"] = sub["urgencia_norm"].replace({"High":"Alta","Medium":"Media","Low":"Baja"})
    ct = pd.crosstab(sub["issue_group"], sub["urgencia_norm"]).reindex(columns=urg_levels, fill_value=0)
    idx = np.arange(len(ct.index))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9,6))
    for i, urg in enumerate(urg_levels):
        vals = ct[urg].values if urg in ct.columns else np.zeros(len(ct.index))
        ax.bar(idx + (i-1)*width, vals, width=width, label=urg)
    ax.set_xticks(idx); ax.set_xticklabels(list(ct.index), rotation=45, ha="right")
    ax.set_title("Distribución de Urgencias en Top Issues"); ax.set_ylabel("Casos"); ax.legend()
    p = os.path.join(out_dir, "urgencia_top_issues.png")
    plt.tight_layout(); fig.savefig(p); plt.close(fig)
    return p, ct

# ===================== Comparativa WoW =====================

def compare_with_prev(issues_df: pd.DataFrame, hist_dir="./hist") -> pd.DataFrame:
    os.makedirs(hist_dir, exist_ok=True)
    today = date.today().isoformat()
    issues_df.to_csv(os.path.join(hist_dir, f"issues_{today}.csv"), index=False, encoding="utf-8")

    prevs = sorted([p for p in os.listdir(hist_dir) if p.startswith("issues_") and p.endswith(".csv")])
    if len(prevs) < 2:
        issues_df["wow_change_pct"] = ""
        issues_df["anomaly_flag"] = False
        return issues_df

    prev = pd.read_csv(os.path.join(hist_dir, prevs[-2]))
    prev_map = dict(zip(prev["issue"], prev["casos"]))

    def calc(issue, casos):
        p = prev_map.get(issue, 0)
        if p == 0:
            return 100.0, True if casos >= 10 else False
        delta = (casos - p) / max(p, 1) * 100.0
        return round(delta, 1), (delta >= 50.0 and casos >= 10)

    issues_df["wow_change_pct"], issues_df["anomaly_flag"] = zip(*issues_df.apply(lambda r: calc(r["issue"], r["casos"]), axis=1))
    return issues_df

# ===================== Gemini (API REST) + presupuesto/cache =====================

class _AIBudget:
    def __init__(self, budget:int):
        self.budget = max(0, int(budget))
        self.rate_limited = False
    def take(self) -> bool:
        if self.rate_limited or self.budget <= 0:
            return False
        self.budget -= 1
        return True

def _dataset_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    counts = df["issue_group"].value_counts().to_dict() if "issue_group" in df.columns else {}
    h.update(json.dumps(counts, sort_keys=True).encode())
    if "created_at" in df.columns:
        try:
            h.update(str(pd.to_numeric(df["created_at"], errors="coerce").sum()).encode())
        except Exception:
            pass
    return h.hexdigest()[:16]

def _load_ai_cache(cache_path: str) -> dict:
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_ai_cache(cache_path: str, cache_obj: dict):
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def gemini_generate_text(prompt, api_key=None, model="gemini-1.5-flash",
                         temperature=0.3, max_output_tokens=256,
                         retries=4, backoff=2.0):
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY/GOOGLE_API_KEY no seteado. Saltando.")
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(data), timeout=40)
            if r.status_code == 429:
                print("❌ Gemini 429 (rate limit).")
                return "__RATE_LIMIT__"
            if r.status_code == 503:
                wait = backoff ** attempt
                print(f"❌ Gemini 503 (overloaded). Retry {attempt+1}/{retries} en {wait:.1f}s")
                time.sleep(wait)
                continue
            if not r.ok:
                print(f"❌ Gemini HTTP {r.status_code}. Body: {r.text[:400]}")
                return "__UNAVAILABLE__"
            out = r.json()
            cand = (out.get("candidates") or [{}])[0]
            parts = ((cand.get("content") or {}).get("parts") or [{}])
            return parts[0].get("text", "").strip()
        except requests.Timeout:
            print("⏱️ Timeout Gemini.")
        except Exception as e:
            print(f"❌ Excepción Gemini: {e}")
    return "__UNAVAILABLE__"

def _gemini_smoke_test(api_key: str | None, model: str) -> None:
    if not (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("ℹ️ Gemini no configurado (sin API key). Insights deshabilitados.")
        return
    txt = gemini_generate_text("Ping: responde 'OK' si recibiste este mensaje.", api_key=api_key, model=model, max_output_tokens=8)
    if txt:
        print(f"✅ Gemini listo. Respuesta: {txt[:50]}")
    else:
        print("⚠️ Gemini no respondió. Verifica key, cuotas o permisos.")

def ai_insight_for_chart(chart_name: str, stats_obj, api_key: str | None = None, model: str = "gemini-1.5-flash") -> str:
    if isinstance(stats_obj, pd.DataFrame):
        snap = stats_obj.head(10).to_string()
    else:
        try:
            snap = json.dumps(stats_obj, ensure_ascii=False)[:4000]
        except Exception:
            snap = str(stats_obj)[:4000]
    prompt = (
        f"Eres un analista de Customer Experience en una empresa de tickets. "
        f"Analiza el gráfico '{chart_name}'. "
        f"Identifica el hallazgo más relevante y su implicancia operativa. "
        f"2 frases máximo, español, tono ejecutivo. Datos:\n{snap}\n"
        "Formato: observación concreta + recomendación."
    )
    return gemini_generate_text(prompt, api_key=api_key, model=model, max_output_tokens=200)

def ai_actions_for_issue(issue: str, contexto: dict, api_key: str | None = None, model: str = "gemini-1.5-flash") -> dict:
    ctx_json = json.dumps(contexto, ensure_ascii=False)
    prompt = (
        f"Eres PM/Analyst en una empresa de tickets. "
        f"Propón 1-3 acciones por categoría para '{issue}': Producto, Tech y CX. "
        "Bullets concretos (≤14 palabras). "
        f"Contexto: {ctx_json}\n"
        "Devuelve SOLO JSON válido con claves 'Producto','Tech','CX'."
    )
    txt = gemini_generate_text(prompt, api_key=api_key, model=model, max_output_tokens=320)
    if txt == "__RATE_LIMIT__":
        return {"__rate_limited__": True}
    try:
        obj = json.loads(txt)
        out = {}
        for k in ["Producto","Tech","CX"]:
            if isinstance(obj.get(k), list):
                out[k] = [str(a) for a in obj[k]][:3]
        return out
    except Exception:
        return {}

# ===================== Publicación en GitHub (assets) =====================

def _run_git(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr or p.stdout)
    return (p.stdout or "").strip()

def _parse_remote_origin(remote_url: str):
    u = remote_url.strip()
    if u.endswith(".git"):
        u = u[:-4]
    if "github.com/" in u:
        owner_repo = u.split("github.com/")[-1]
    elif "github.com:" in u:
        owner_repo = u.split("github.com:")[-1]
    else:
        raise ValueError(f"No pude parsear remote.origin.url: {remote_url}")
    parts = u.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    raise ValueError(f"URL inesperada: {remote_url}")

def publish_images_to_github(out_dir: str,
                             repo_path: str,
                             branch: str = "main",
                             date_subdir: str | None = None,
                             files: list[str] | None = None) -> str:
    if files is None:
        files = ["top_issues.png", "urgencia_pie.png", "sentimiento_pie.png",
                 "urgencia_por_issue.png", "canal_por_issue.png", "urgencia_top_issues.png"]
    if date_subdir is None:
        date_subdir = date.today().isoformat()

    if not os.path.isdir(repo_path):
        raise RuntimeError(f"Repo path no existe: {repo_path}")
    for fn in files:
        p = os.path.join(out_dir, fn)
        if not os.path.exists(p):
            raise RuntimeError(f"No existe imagen: {p}")

    dest_dir = os.path.join(repo_path, "reports", date_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    for fn in files:
        shutil.copy2(os.path.join(out_dir, fn), os.path.join(dest_dir, fn))

    _run_git(["git", "add", "."], cwd=repo_path)
    try:
        _run_git(["git", "commit", "-m", f"Report {date_subdir}: PNG charts"], cwd=repo_path)
    except RuntimeError as e:
        if "nothing to commit" not in str(e).lower():
            raise
    _run_git(["git", "push", "origin", branch], cwd=repo_path)

    remote = _run_git(["git", "config", "--get", "remote.origin.url"], cwd=repo_path)
    owner, repo = _parse_remote_origin(remote)
    base_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/reports/{date_subdir}"
    return base_raw

# ===================== Notion helpers (bloques + URL sanitizer) =====================

def _h1(text): return {"object":"block","type":"heading_1","heading_1":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def _h2(text): return {"object":"block","type":"heading_2","heading_2":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def _h3(text): return {"object":"block","type":"heading_3","heading_3":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def _para(text): return {"object":"block","type":"paragraph","paragraph":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def _bullet(text): return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":[{"type":"text","text":{"content":text}}]}}
def _todo(text, checked=False): return {"object":"block","type":"to_do","to_do":{"rich_text":[{"type":"text","text":{"content":text}}],"checked":checked}}
def _callout(text, icon="💡"): return {"object":"block","type":"callout","callout":{"rich_text":[{"type":"text","text":{"content":text}}],"icon":{"type":"emoji","emoji":icon}}}
def _divider(): return {"object":"block","type":"divider","divider":{}}
def _rt(text: str): return [{"type": "text", "text": {"content": str(text)}}]

URL_RX = re.compile(r'^https?://[^\s<>"\'\|\)\]]+$', re.IGNORECASE)

def _clean_url(u: str) -> str | None:
    if not isinstance(u, str): return None
    u = u.strip().replace("\n", " ").replace("\r", " ")
    if not u: return None
    u = u.split()[0].strip('\'"()[]')
    u = ''.join(ch for ch in u if 31 < ord(ch) < 127 and ch not in {'|'})
    if not (u.lower().startswith("http://") or u.lower().startswith("https://")):
        return None
    return u if URL_RX.match(u) else None

def _link(text: str, url: str):
    safe = _clean_url(url)
    if not safe:
        return {"type":"text","text":{"content":str(text)}}
    return {"type":"text","text":{"content":str(text),"link":{"url":safe}}}

def _image_external_if_valid(url: str | None, caption: str | None = None):
    safe = _clean_url(url) if url else None
    if not safe:
        return None
    b = {"object":"block","type":"image","image":{"type":"external","external":{"url":safe}}}
    if caption:
        b["image"]["caption"] = [{"type":"text","text":{"content":caption}}]
    return b

def _column_list(columns_children: list[list[dict]]):
    return {"object":"block","type":"column_list","column_list":{"children":[{"object":"block","type":"column","column":{"children":ch}} for ch in columns_children]}}

def _notion_table(headers: list[str], rows: list[list[list[dict]]]):
    table = {"object":"block","type":"table","table":{"table_width": len(headers),"has_column_header": True,"has_row_header": False,"children":[]}}
    table["table"]["children"].append({"object":"block","type":"table_row","table_row":{"cells":[_rt(h) for h in headers]}})
    for row in rows:
        table["table"]["children"].append({"object":"block","type":"table_row","table_row":{"cells": row}})
    return table

def build_issues_table_block(resumen_df: pd.DataFrame) -> dict:
    headers = ["Issue","Casos","Canales","Areas","Motivos","Submotivos","Ejemplos"]
    rows = []
    for _, r in resumen_df.sort_values("casos", ascending=False).iterrows():
        ej_text = str(r.get("ejemplos_intercom","") or "")
        parts = [p.strip() for p in ej_text.replace("\n"," ").split("|") if p.strip()]
        urls = []
        seen = set()
        for p in parts:
            u = _clean_url(p)
            if u and u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= 3:
                break
        examples_rt = [{"type":"text","text":{"content":"—"}}] if not urls else sum(
            [[_link(f"Ejemplo {i}", u)] + ([{"type":"text","text":{"content":"  •  "}}] if i < len(urls) else []) for i,u in enumerate(urls,1)], [])
        row_cells = [
            _rt(r.get("issue","")),
            _rt(str(int(r.get("casos",0) or 0))),
            _rt(r.get("canales_top","")),
            _rt(r.get("areas_top","")),
            _rt(r.get("motivos_top","")),
            _rt(r.get("submotivos_top","")),
            examples_rt
        ]
        rows.append(row_cells)
    return _notion_table(headers, rows)

ACTION_LIBRARY = {
    "Pagos / cobros": {
        "Producto": ["Mostrar causa de rechazo y reintentos guiados.","Guardar tarjeta segura (1-click)."],
        "Tech": ["Logs de PSP + alertas por BIN/issuer.","Retries con backoff en fallas de red."],
        "CX": ["Macro paso a paso por medio de pago.","Comunicar reserva temporal."]
    },
    "Entrega de entradas": {
        "Producto": ["CTA visible para reenvío y confirmación en UI."],
        "Tech": ["Job idempotente de reenvío.","Monitoreo de bounce/spam."],
        "CX": ["Bot autogestivo de reenvío por mail/WhatsApp."]
    },
    "QR / Validación en acceso": {
        "Producto": ["Feedback claro de estado del QR (válido/usado/bloqueado)."],
        "Tech": ["Telemetría de validadores + health checks."],
        "CX": ["Guía de acceso y resolución de errores comunes."]
    },
    "Transferencia / titularidad": {
        "Producto": ["Flujo guiado de cambio de titularidad con confirmación."],
        "Tech": ["Auditoría/registro de transferencias."],
        "CX": ["Macro con costos/plazos y límites."]
    },
    "Reembolso / devolución": {
        "Producto": ["Estado visible del reembolso y tiempos estimados."],
        "Tech": ["Idempotencia + conciliación con PSP."],
        "CX": ["Macro de seguimiento y expectativas."]
    }
}

def actions_for_issue(issue: str):
    return ACTION_LIBRARY.get(issue, {
        "Producto": ["Quick wins de UX para reducir fricción."],
        "Tech": ["Registrar error types + tracing/dashboards."],
        "CX": ["Macro de contención + FAQ específica."]
    })

def build_actions_section_blocks(resumen_df: pd.DataFrame, top_n: int = 5, acciones_ai: dict | None = None) -> list[dict]:
    blocks = []
    blocks.append(_h2("Acciones A Evaluar"))
    for _, r in resumen_df.sort_values("casos", ascending=False).head(top_n).iterrows():
        issue = str(r.get("issue",""))
        casos = int(r.get("casos",0) or 0)
        blocks.append(_h3(f"{issue} ({casos})"))
        base = actions_for_issue(issue)
        ai_extra = acciones_ai.get(issue, {}) if acciones_ai else {}
        col_prod = [ _para("Producto:") ] + [ _todo(a) for a in base.get("Producto", []) + ai_extra.get("Producto", []) ]
        col_tech = [ _para("Tech:") ]     + [ _todo(a) for a in base.get("Tech", []) + ai_extra.get("Tech", []) ]
        col_cx   = [ _para("CX:") ]       + [ _todo(a) for a in base.get("CX", []) + ai_extra.get("CX", []) ]
        blocks.append(_column_list([col_prod, col_tech, col_cx]))
    return blocks

# -------- sanitizador transversal de links --------

def _sanitize_links_in_blocks(blocks: list[dict]) -> list[dict]:
    blks = json.loads(json.dumps(blocks))
    def strip_or_fix_rt(rt_list):
        for tkn in rt_list:
            if tkn.get("type") == "text":
                link = tkn.get("text", {}).get("link")
                if link and "url" in link:
                    safe = _clean_url(link["url"])
                    if not safe:
                        tkn["text"]["link"] = None
                    else:
                        tkn["text"]["link"]["url"] = safe
        return rt_list
    for b in blks:
        t = b.get("type")
        if t in ("paragraph","bulleted_list_item","to_do","heading_1","heading_2","heading_3","callout"):
            if t in b and "rich_text" in b[t]:
                b[t]["rich_text"] = strip_or_fix_rt(b[t]["rich_text"])
        elif t == "table":
            for row in b.get("table", {}).get("children", []):
                cells = row.get("table_row", {}).get("cells", [])
                for i, cell in enumerate(cells):
                    cells[i] = strip_or_fix_rt(cell)
        elif t == "image":
            img = b.get("image", {})
            if img.get("type") == "external" and "external" in img and "url" in img["external"]:
                safe = _clean_url(img["external"]["url"])
                if not safe:
                    caption = ""
                    cap_rt = img.get("caption") or []
                    if cap_rt and cap_rt[0].get("type") == "text":
                        caption = cap_rt[0]["text"].get("content","")
                    b.clear(); b.update(_para(caption or ""))
                else:
                    img["external"]["url"] = safe
            cap = img.get("caption", [])
            if cap:
                b["image"]["caption"] = strip_or_fix_rt(cap)
    return blks

# ---------- KPIs UI helpers ----------

def _metric_card(title: str, value: str, sub: str = "", icon: str = "📊") -> dict:
    text = f"{title}: {value}" if not sub else f"{title}: {value}\n{sub}"
    return _callout(text, icon=icon)

def build_kpi_section_blocks(kpis: dict | None, total_items: int) -> list[dict]:
    blocks = []
    blocks.append(_h2("KPIs a la vista"))
    if not kpis:
        blocks.append(_para("—"))
        return blocks

    tickets = str(kpis.get("tickets_resueltos","—"))
    pct_val = kpis.get("tasa_1ra_respuesta", None)
    pct = f"{pct_val:.0f}%" if isinstance(pct_val, (int,float)) else "—"
    base = kpis.get("first_base", 0); ok = kpis.get("first_ok", 0)
    med_first = _fmt_min_from_sec(kpis.get("first_resp_p50_s"))
    ttr_mean = _fmt_h(kpis.get("ttr_horas"))
    ttr_p50  = _fmt_h(kpis.get("ttr_p50_h"))
    ttr_p90  = _fmt_h(kpis.get("ttr_p90_h"))
    csat = kpis.get("csat", None); csn = kpis.get("csat_n",0)
    csat_txt = "—" if csat is None else f"{csat:.2f}"
    sla = kpis.get("sla_minutes", 15)

    row1 = [
        [_metric_card("Tickets resueltos", tickets, f"Total convers. procesadas: {total_items}", "🎟️")],
        [_metric_card("% 1ra respuesta en SLA", pct, f"SLA {sla} min • {ok}/{base}", "⚡")],
        [_metric_card("TTR medio", ttr_mean, f"p50 {ttr_p50} • p90 {ttr_p90}", "⏱️")],
        [_metric_card("CSAT", csat_txt, f"n={csn}", "⭐")],
    ]
    blocks.append(_column_list([c for c in row1]))

    blocks.append(_divider())
    blocks.append(_bullet(f"Tickets resueltos: {tickets}"))
    blocks.append(_bullet(f"% 1ra respuesta: {pct} ({ok}/{base}); mediana {med_first}"))
    blocks.append(_bullet(f"Tiempo medio de resolución: {ttr_mean} (p50 {ttr_p50} • p90 {ttr_p90})"))
    blocks.append(_bullet(f"Satisfacción (CSAT): {('—' if csat is None else f'{csat_txt}')} (n={csn})"))
    return blocks

# ===================== Página Notion =====================

def notion_create_page(parent_page_id: str,
                       token: str,
                       page_title: str,
                       df: pd.DataFrame,
                       resumen_df: pd.DataFrame,
                       meta: dict,
                       chart_urls: dict,
                       insights: dict,
                       acciones_ai: dict | None = None,
                       kpis: dict | None = None):
    blocks = []
    blocks.append(_h1(page_title))

    # Resumen
    blocks.append(_h2("Resumen Ejecutivo"))
    blocks.append(_para(f"📅 Fecha del análisis: {meta.get('fecha','')}"))
    blocks.append(_para(f"📂 Fuente de datos: {meta.get('fuente','')}"))
    blocks.append(_para(f"💬 Conversaciones procesadas: {meta.get('total','')}"))
    blocks.append(_para("Durante el periodo analizado se registraron conversaciones en Intercom, procesadas por IA para identificar patrones, problemas recurrentes y oportunidades de mejora."))
    # Foco ejecutivo (nuevo)
    blocks.append(_callout("Foco de la semana: reducir TTR p50 en Top-2 issues (-20% en 14 días) y +0.2 CSAT.", icon="🎯"))

    # KPIs
    blocks.extend(build_kpi_section_blocks(kpis, total_items=len(df)))

    # Top 3 issues
    blocks.append(_h2("Top 3 issues"))
    total_len = max(len(df), 1)
    top3 = df["issue_group"].value_counts().head(3)
    if len(top3) == 0:
        blocks.append(_para("—"))
    else:
        for issue, casos in top3.items():
            pct_issue = f"{(casos/total_len*100):.0f}%"
            blocks.append(_bullet(f"{issue} → {casos} casos ({pct_issue})"))
    if insights.get("top_issues"):
        blocks.append(_callout(insights["top_issues"], icon="💡"))

    # Gráficos + insights
    for key, caption in [
        ("top_issues", "Top Issues"),
        ("urgencia_pie", "Distribución de Urgencias"),
        ("sentimiento_pie", "Distribución de Sentimientos"),
        ("urgencia_por_issue", "Urgencia por Issue"),
        ("urgencia_top_issues", "Urgencias en Top Issues (agrupadas)"),
        ("canal_por_issue", "Canal por Issue"),
    ]:
        blk = _image_external_if_valid((chart_urls or {}).get(key), caption)
        if blk:
            blocks.append(blk)
        if insights.get(key):
            blocks.append(_callout(insights[key], icon="💡"))

    # Categorizaciones manuales
    blocks.append(_h2("Categorizaciones manuales"))

    # Tema
    blocks.append(_h3("Tema"))
    tema_series = df["tema_norm"].fillna("").replace({"nan": ""}).astype(str).str.strip()
    tema_vc = tema_series[tema_series != ""].value_counts().head(5)
    if len(tema_vc) == 0:
        blocks.append(_para("—"))
    else:
        for k, v in tema_vc.items():
            blocks.append(_bullet(f"{k}: {v}"))
    if insights.get("tema_counts"):
        blocks.append(_callout(insights["tema_counts"], icon="💡"))

    # Motivo
    blocks.append(_h3("Motivo"))
    motivo_series = df["motivo_norm"].fillna("").replace({"nan": ""}).astype(str).str.strip()
    motivo_vc = motivo_series[motivo_series != ""].value_counts().head(5)
    if len(motivo_vc) == 0:
        blocks.append(_para("—"))
    else:
        for k, v in motivo_vc.items():
            blocks.append(_bullet(f"{k}: {v}"))
    if insights.get("motivo_counts"):
        blocks.append(_callout(insights["motivo_counts"], icon="💡"))

    # Issues Detallados (tabla)
    blocks.append(_h2("Issues Detallados"))
    blocks.append(build_issues_table_block(resumen_df))

    # Acciones a evaluar
    blocks.extend(build_actions_section_blocks(resumen_df, top_n=5, acciones_ai=acciones_ai))

    # Sanitizado final de enlaces
    safe_children = _sanitize_links_in_blocks(blocks)

    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "properties": {"title": {"title": [{"text": {"content": page_title}}]}},
        "children": safe_children
    }

    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        },
        data=json.dumps(payload)
    )

    # Retry sin links si Notion protesta por URLs inválidas
    if not resp.ok and ("Invalid URL" in (resp.text or "") or "link" in (resp.text or "").lower()):
        def strip_all_links(blks: list[dict]) -> list[dict]:
            blks = json.loads(json.dumps(blks))
            for b in blks:
                t = b.get("type")
                if t in ("paragraph", "bulleted_list_item", "to_do", "heading_1", "heading_2", "heading_3", "callout"):
                    rt = b.get(t, {}).get("rich_text", [])
                    for tkn in rt:
                        if tkn.get("type") == "text" and tkn.get("text", {}).get("link"):
                            tkn["text"]["link"] = None
                elif t == "table":
                    for row in b.get("table", {}).get("children", []):
                        cells = row.get("table_row", {}).get("cells", [])
                        for cell in cells:
                            for tkn in cell:
                                if tkn.get("type") == "text" and tkn.get("text", {}).get("link"):
                                    tkn["text"]["link"] = None
                elif t == "image":
                    cap = b.get("image", {}).get("caption", [])
                    for tkn in cap:
                        if tkn.get("type") == "text" and tkn.get("text", {}).get("link"):
                            tkn["text"]["link"] = None
            return blks

        payload_retry = dict(payload)
        payload_retry["children"] = strip_all_links(payload["children"])
        resp = requests.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload_retry)
        )

    if not resp.ok:
        raise RuntimeError(f"Notion error {resp.status_code}: {resp.text}")
    return resp.json()

# ===================== Core =====================

def run(csv_path: str,
        out_dir: str,
        notion_token: str | None,
        notion_parent: str | None,
        publish_github: bool = False,
        github_repo_path: str | None = None,
        github_branch: str = "main",
        assets_base_url: str | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-1.5-flash",
        sla_first_reply_min: int = 15,
        ai_mode: str = "full",
        ai_budget: int = 100):

    os.makedirs(out_dir, exist_ok=True)

    # Lectura + normalización + taxonomías
    df = load_csv_robusto(csv_path)
    df = normalize_columns(df)
    for col in ["tema","motivo","submotivo","urgencia","canal","area","sentimiento","categoria"]:
        if col in df.columns:
            df[col] = df[col].astype(object)
    df = enforce_taxonomy(df)

    # ---------------- NUEVO: consolidar csat / estado_final / issue estable ----------------
    df["csat_num"] = df.apply(pick_csats, axis=1)
    if "csat" in df.columns:
        df["csat"] = df["csat"].where(pd.to_numeric(df["csat"], errors="coerce").between(1,5), df["csat_num"])
    else:
        df["csat"] = df["csat_num"]

    df["estado_final"] = df.apply(estado_final_from, axis=1)

    if "texto_base" not in df.columns:
        df["texto_base"] = df.apply(build_text_base, axis=1)
        
    for c in ["motivo_norm", "submotivo_norm", "texto_base"]:
        if c in df.columns:
            df[c] = df[c].astype(object).where(~pd.isna(df[c]), "")

    df["issue_group"] = df.apply(
        lambda r: choose_issue(r.get("motivo_norm"), r.get("submotivo_norm"), r.get("texto_base")),
        axis=1
    )
    # -------------------------------------------------------------------

    # Prueba rápida de Gemini
    _gemini_smoke_test(gemini_api_key, gemini_model)

    # KPI desde crudos
    kpis = compute_kpis_from_raw_df(df, sla_minutes=sla_first_reply_min)

    # Reforzar "tickets resueltos" por estado_final
    if "estado_final" in df.columns:
        kpis["tickets_resueltos"] = int((df["estado_final"] == "Resuelto").sum())

    # Issue grouping (mantengo el original por si lo usabas para contrastar)
    # df["issue_group"] ya set por choose_issue()

    # Flags
    flags = df.apply(compute_flags, axis=1)
    for c in flags.columns:
        df[c] = flags[c]

    # Resumen por issue
    rows = []
    for issue, grp in df.groupby("issue_group"):
        canales = top_values(grp["canal"])
        areas = top_values(grp["area"])
        motivos = top_values(grp["motivo_norm"])
        submotivos = top_values(grp["submotivo_norm"])
        ejemplos = grp.loc[grp["link_a_intercom"].astype(str).str.len() > 0, "link_a_intercom"].head(3).astype(str).tolist()
        rows.append({
            "issue": issue,
            "casos": int(len(grp)),
            "canales_top": canales,
            "areas_top": areas,
            "motivos_top": motivos,
            "submotivos_top": submotivos,
            "ejemplos_intercom": " | ".join(ejemplos)
        })
    resumen_df = pd.DataFrame(rows).sort_values("casos", ascending=False)
    resumen_df = compare_with_prev(resumen_df, hist_dir=os.path.join(out_dir, "hist"))

    # Exports CSV
    issues_csv = os.path.join(out_dir, "issues_resumen.csv")
    casos_csv = os.path.join(out_dir, "casos_con_issue.csv")
    df_export_cols = ["fecha","canal","rol","area","tema_norm","motivo_norm","submotivo_norm",
                      "categoria","urgencia","sentimiento","resumen_ia","insight_ia",
                      "palabras_clave","issue_group","estado_final","risk","sla_now","taxonomy_flag",
                      "link_a_intercom","id_intercom",
                      "created_at","first_admin_reply_at","first_contact_reply_at","closed_at",
                      "first_response_seconds","ttr_seconds","status","csat","csat_num"]
    existing = [c for c in df_export_cols if c in df.columns]
    df[existing].to_csv(casos_csv, index=False, encoding="utf-8")
    resumen_df.to_csv(issues_csv, index=False, encoding="utf-8")

    total = len(df)

    # Gráficos
    p_top, top_counts = chart_top_issues(df, out_dir)
    p_urg_pie, urg_counts = chart_urgencia_pie(df, out_dir)
    p_sent_pie, sent_counts = chart_sentimiento_pie(df, out_dir)
    p_urg_issue, urg_issue_ct = chart_urgencia_por_issue(df, out_dir)
    p_canal_issue, canal_issue_ct = chart_canal_por_issue(df, out_dir)
    p_urg_top, urg_top_ct = chart_urgencias_en_top_issues(df, out_dir)

    # Assets públicos
    if publish_github and github_repo_path:
        try:
            assets_base_url = publish_images_to_github(
                out_dir=out_dir,
                repo_path=github_repo_path,
                branch=github_branch,
                date_subdir=date.today().isoformat(),
                files=[
                    os.path.basename(p_top),
                    os.path.basename(p_urg_pie),
                    os.path.basename(p_sent_pie),
                    os.path.basename(p_urg_issue),
                    os.path.basename(p_canal_issue),
                    os.path.basename(p_urg_top),
                ],
            )
            print(f"🌐 Assets publicados en: {assets_base_url}")
        except Exception as e:
            print(f"⚠️ No pude publicar en GitHub: {e}")

    chart_urls = {}
    if assets_base_url:
        chart_urls = {
            "top_issues": f"{assets_base_url}/{os.path.basename(p_top)}",
            "urgencia_pie": f"{assets_base_url}/{os.path.basename(p_urg_pie)}",
            "sentimiento_pie": f"{assets_base_url}/{os.path.basename(p_sent_pie)}",
            "urgencia_por_issue": f"{assets_base_url}/{os.path.basename(p_urg_issue)}",
            "canal_por_issue": f"{assets_base_url}/{os.path.basename(p_canal_issue)}",
            "urgencia_top_issues": f"{assets_base_url}/{os.path.basename(p_urg_top)}",
        }

    # ===== IA: modos, presupuesto y cache =====
    mode = (ai_mode or "full").strip().lower()
    if mode == "off":
        budget = 0
    elif mode == "lite":
        budget = min(ai_budget, 6) if ai_budget else 6
    else:
        budget = ai_budget if ai_budget else 100

    ai = _AIBudget(budget)
    cache_path = os.path.join(out_dir, "hist", "ai_cache.json")
    cache = _load_ai_cache(cache_path)
    fp = _dataset_fingerprint(df)

    insights = {}
    acciones_ai = {}

    if fp in cache:
        cached = cache[fp]
        insights = cached.get("insights", {})
        acciones_ai = cached.get("acciones_ai", {})
        print(f"♻️ Reutilizando insights desde cache. insights={len(insights)} | acciones={len(acciones_ai)}")
    elif budget > 0 and (gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        def try_insight(name, obj):
            nonlocal insights
            if not ai.take():
                return
            txt = ai_insight_for_chart(name, obj, api_key=gemini_api_key, model=gemini_model)
            if txt == "__RATE_LIMIT__":
                ai.rate_limited = True
                return
            if txt:
                insights[name] = txt

        try_insight("top_issues", top_counts.to_dict())
        try_insight("urgencia_pie", urg_counts.to_dict())
        try_insight("sentimiento_pie", sent_counts.to_dict())
        if not ai.rate_limited:
            try_insight("urgencia_por_issue", urg_issue_ct)
        if not ai.rate_limited:
            try_insight("canal_por_issue", canal_issue_ct)
        if not ai.rate_limited:
            try_insight("urgencia_top_issues", urg_top_ct)

        if not ai.rate_limited and (mode == "full" or ai.budget >= 2):
            tema_counts = df["tema_norm"].fillna("").replace({"nan":""}).astype(str).str.strip()
            tema_counts = tema_counts[tema_counts != ""].value_counts().head(5)
            if ai.take():
                ttxt = ai_insight_for_chart("top_temas", tema_counts.to_dict(), api_key=gemini_api_key, model=gemini_model)
                if ttxt and ttxt != "__RATE_LIMIT__":
                    insights["tema_counts"] = ttxt
                elif ttxt == "__RATE_LIMIT__":
                    ai.rate_limited = True

            if not ai.rate_limited and ai.take():
                motivo_counts = df["motivo_norm"].fillna("").replace({"nan":""}).astype(str).str.strip()
                motivo_counts = motivo_counts[motivo_counts != ""].value_counts().head(5)
                mtxt = ai_insight_for_chart("top_motivos", motivo_counts.to_dict(), api_key=gemini_api_key, model=gemini_model)
                if mtxt and mtxt != "__RATE_LIMIT__":
                    insights["motivo_counts"] = mtxt
                elif mtxt == "__RATE_LIMIT__":
                    ai.rate_limited = True

        max_actions = 5 if mode == "full" else 2 if mode == "lite" else 0
        for _, row in resumen_df.head(max_actions).iterrows():
            if ai.rate_limited or not ai.take():
                break
            issue = str(row.get("issue",""))
            ctx = {
                "canales_top": str(row.get("canales_top","")),
                "areas_top": str(row.get("areas_top","")),
                "motivos_top": str(row.get("motivos_top","")),
                "submotivos_top": str(row.get("submotivos_top","")),
                "total_casos_issue": int(row.get("casos",0) or 0),
                "total_casos_semana": total
            }
            add = ai_actions_for_issue(issue, ctx, api_key=gemini_api_key, model=gemini_model)
            if add.get("__rate_limited__"):
                ai.rate_limited = True
                break
            if add:
                acciones_ai[issue] = add

        if insights or acciones_ai:
            cache[fp] = {"insights": insights, "acciones_ai": acciones_ai}
            _save_ai_cache(cache_path, cache)

    used = budget - ai.budget
    print(f"🤖 IA → modo={mode} | usadas={used}/{budget} | rate_limited={ai.rate_limited} | insights={len(insights)} | acciones={len(acciones_ai)}")

    # Notion
    if notion_token and notion_parent:
        page_title = f"Reporte CX – {date.today().isoformat()}"
        meta = {"fecha": date.today().isoformat(), "fuente": os.path.basename(csv_path), "total": total}
        page = notion_create_page(
            parent_page_id=notion_parent,
            token=notion_token,
            page_title=page_title,
            df=df,
            resumen_df=resumen_df,
            meta=meta,
            chart_urls=chart_urls,
            insights=insights,
            acciones_ai=acciones_ai if acciones_ai else None,
            kpis=kpis
        )
        print(f"✅ Publicado en Notion: {page.get('url','(sin url)')}")
    else:
        print("ℹ️ Notion no configurado. Se generaron archivos locales.")
        print(json.dumps({
            "issues_csv": issues_csv,
            "casos_csv": casos_csv,
            "charts": {
                "top_issues": p_top,
                "urgencia_pie": p_urg_pie,
                "sentimiento_pie": p_sent_pie,
                "urgencia_por_issue": p_urg_issue,
                "canal_por_issue": p_canal_issue,
                "urgencia_top_issues": p_urg_top
            },
            "insights": insights
        }, indent=2, ensure_ascii=False))

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Venti – Insight IA (Notion narrativo + Gemini API)")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de conversaciones")
    ap.add_argument("--out", default="./salida", help="Directorio de salida")
    ap.add_argument("--notion_token", default=os.getenv("NOTION_TOKEN"), help="Token de Notion")
    ap.add_argument("--notion_parent", default=os.getenv("NOTION_PARENT_PAGE_ID"), help="ID de página padre en Notion")
    ap.add_argument("--publish_github", action="store_true", help="Publicar PNGs al repo y usar URL pública")
    ap.add_argument("--github_repo_path", default=None, help="Ruta local al repo clonado")
    ap.add_argument("--github_branch", default="main", help="Branch destino")
    ap.add_argument("--assets_base_url", default=None, help="URL base pública ya hosteada (si no publicas a GitHub)")
    ap.add_argument("--gemini_api_key", default=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"), help="API key de Gemini")
    ap.add_argument("--gemini_model", default="gemini-1.5-flash", help="Modelo de Gemini")
    ap.add_argument("--sla_first_reply_min", type=int, default=15, help="SLA (min) para % 1ra respuesta")
    ap.add_argument("--ai_mode", choices=["full","lite","off"], default="full", help="Nivel de generación con IA")
    ap.add_argument("--ai_budget", type=int, default=100, help="Máximo de requests a Gemini por corrida")
    args = ap.parse_args()

    print("▶ Script iniciado")
    print("CSV:", args.csv)
    print("OUT:", args.out)
    print("NOTION_TOKEN:", "OK" if args.notion_token else "FALTA")
    print("PARENT:", args.notion_parent)

    run(
        csv_path=args.csv,
        out_dir=args.out,
        notion_token=args.notion_token,
        notion_parent=args.notion_parent,
        publish_github=args.publish_github,
        github_repo_path=args.github_repo_path,
        github_branch=args.github_branch,
        assets_base_url=args.assets_base_url,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        sla_first_reply_min=args.sla_first_reply_min,
        ai_mode=args.ai_mode,
        ai_budget=args.ai_budget
    )

if __name__ == "__main__":
    main()