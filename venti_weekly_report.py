# -*- coding: utf-8 -*-
"""
Venti – Insight IA: Reporte semanal Intercom (Notion narrativo + Gemini API)
"""

import os
import re
import json
import argparse
import subprocess
import shutil
from datetime import date
import requests
import pandas as pd
import numpy as np

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== Utilidades base =====================

EMOJI_RX = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)

def strip_emojis(s: str) -> str:
    try:
        return EMOJI_RX.sub("", s or "")
    except Exception:
        return s or ""

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
        cc = (cc.replace("í", "i").replace("á", "a").replace("é", "e")
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

    for canon in ["resumen_ia","insight_ia","palabras_clave","canal","area","tema",
                  "motivo","submotivo","urgencia","sentimiento","categoria",
                  "link_a_intercom","id_intercom","fecha","rol"]:
        if canon not in df.columns:
            df[canon] = ""
    return df

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
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "", False
        v = str(value).strip().lower()
    except Exception:
        v = ""
    if not v or v in ("nan","none"):
        return "", False
    if v in catalog:
        return v, True
    for item in catalog:
        if v in item or item in v:
            return item, True
    return v, False

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
        val = str(row.get(col,"") or "").strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return " | ".join(parts).lower()

RULES = [
    ("Entrega de entradas", r"(no\s*recib[ií]|reenv(i|í)a|link\s*de\s*entrada|entrada(s)?\s*(no)?\s*llega|ticket\s*no\s|no\s*me\s*lleg[oó])"),
    ("Transferencia / titularidad", r"(transferenc|transferir|cambio\s*de\s*titular|modificar\s*(nombre|titular)|pasar\s*entrada)"),
    ("QR / Validación en acceso", r"(qr|validaci[oó]n|control\s*de\s*acceso|escaneo|lector|validador)"),
    ("Pagos / cobros", r"(pago|pagos|cobro|cobrar|rechazad|tarjeta|mercadopago|mp|cuotas)"),
    ("Reembolso / devolución", r"(reembolso|devoluci[oó]n|refund|chargeback)"),
    ("Cuenta / login / registro", r"(cuenta|logue|login|registr|contrasen(?:a|ñ)|clave|verificaci[oó]n\s*de\s*mail|correo\s*inv[aá]lido)"),
    ("App / rendimiento / bug", r"(app|aplicaci[oó]n|crash|se\s*cierra|no\s*funciona|bug|error\s*(tecnico|500|404))"),
    ("Soporte / sin respuesta / SDU", r"(sin\s*respuesta|derivaci[oó]n\s*al\s*sdu|sdu|jotform|demora|espera)"),
    ("Información de evento", r"(consulta\s*por\s*evento|horario|ubicaci[oó]n|vip|mapa|line\s*up|capacidad|ingreso|puerta)"),
    ("Productores / RRPP / invitaciones", r"(invitaci[oó]n|rrpp|productor|productora|acceso\s*productor|carga\s*de\s*evento|validadores|operativo)"),
]

def assign_issue_group(text: str) -> str:
    for label, pattern in RULES:
        if re.search(pattern, text):
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
    if any(k in txt for k in ["estafa","fraude","no puedo entrar","no puedo ingresar","rechazad"]):
        risk = "HIGH"
    elif urg in ("alta","high"):
        risk = "HIGH"
    elif urg in ("media","medium"):
        risk = "MEDIUM"
    sla_now = (canal in ("whatsapp","instagram")) and (risk == "HIGH")
    return pd.Series({"risk": risk, "sla_now": sla_now})

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

# ===================== Gemini (API REST) =====================

def gemini_generate_text(prompt: str,
                         api_key: str | None = None,
                         model: str = "gemini-1.5-flash",
                         temperature: float = 0.3,
                         max_output_tokens: int = 256) -> str:
    api_key = api_key or os.getenv("AIzaSyBVCbzd3mAIu9k1O4TU5x9ij9nOnMSaUlE") or os.getenv("AIzaSyBVCbzd3mAIu9k1O4TU5x9ij9nOnMSaUlE")
    if not api_key:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(data), timeout=40)
        if not r.ok:
            return ""
        out = r.json()
        cand = (out.get("candidates") or [{}])[0]
        parts = ((cand.get("content") or {}).get("parts") or [{}])
        text = parts[0].get("text", "").strip()
        return text
    except Exception:
        return ""

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
        "Bullets concretos (≤14 palabras), sin relleno ni repeticiones. "
        f"Contexto: {ctx_json}\n"
        "Devuelve SOLO JSON válido con claves 'Producto','Tech','CX'."
    )
    txt = gemini_generate_text(prompt, api_key=api_key, model=model, max_output_tokens=320)
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
    parts = owner_repo.split("/")
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

def _h1(text):
    return {"object":"block","type":"heading_1","heading_1":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def _h2(text):
    return {"object":"block","type":"heading_2","heading_2":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def _h3(text):
    return {"object":"block","type":"heading_3","heading_3":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def _para(text):
    return {"object":"block","type":"paragraph","paragraph":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def _bullet(text):
    return {"object":"block","type":"bulleted_list_item","bulleted_list_item":{"rich_text":[{"type":"text","text":{"content":text}}]}}

def _todo(text, checked=False):
    return {"object":"block","type":"to_do","to_do":{"rich_text":[{"type":"text","text":{"content":text}}],"checked":checked}}

def _callout(text, icon="💡"):
    return {"object":"block","type":"callout","callout":{"rich_text":[{"type":"text","text":{"content":text}}],"icon":{"type":"emoji","emoji":icon}}}

def _rt(text: str):
    return [{"type": "text", "text": {"content": str(text)}}]

URL_RX = re.compile(r'^https?://[^\s<>"\'\|\)\]]+$', re.IGNORECASE)

def _clean_url(u: str) -> str | None:
    if not isinstance(u, str):
        return None
    u = u.strip().replace("\n", " ").replace("\r", " ")
    if not u:
        return None
    # primer token, sin envoltorios
    u = u.split()[0].strip('\'"()[]')
    # quitar caracteres de control / pipes
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
    return {
        "object":"block",
        "type":"column_list",
        "column_list":{"children":[{"object":"block","type":"column","column":{"children":ch}} for ch in columns_children]}
    }

def _notion_table(headers: list[str], rows: list[list[list[dict]]]):
    table = {
        "object":"block",
        "type":"table",
        "table":{
            "table_width": len(headers),
            "has_column_header": True,
            "has_row_header": False,
            "children":[]
        }
    }
    table["table"]["children"].append({
        "object":"block","type":"table_row",
        "table_row":{"cells":[_rt(h) for h in headers]}
    })
    for row in rows:
        table["table"]["children"].append({
            "object":"block","type":"table_row",
            "table_row":{"cells": row}
        })
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
        # Rich text de ejemplos
        if urls:
            examples_rt = []
            for i, u in enumerate(urls, start=1):
                examples_rt.append(_link(f"Ejemplo {i}", u))
                if i < len(urls):
                    examples_rt.append({"type":"text","text":{"content":"  •  "}})
        else:
            examples_rt = [{"type":"text","text":{"content":"—"}}]

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

# -------- sanitizador transversal de links en TODOS los bloques --------

def _sanitize_links_in_blocks(blocks: list[dict]) -> list[dict]:
    """
    Recorre todos los rich_text, celdas de tablas e imágenes y elimina link.href
    cuando la URL no pasa _clean_url. Devuelve una copia segura.
    """
    blks = json.loads(json.dumps(blocks))  # deep copy

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
                    # invalid image URL → convertimos a párrafo plano con el caption (evita 400)
                    caption = ""
                    cap_rt = img.get("caption") or []
                    if cap_rt and cap_rt[0].get("type") == "text":
                        caption = cap_rt[0]["text"].get("content","")
                    b.clear()
                    b.update(_para(caption or ""))
                else:
                    img["external"]["url"] = safe
            # también saneamos el caption
            cap = img.get("caption", [])
            if cap:
                b["image"]["caption"] = strip_or_fix_rt(cap)

    return blks

# ===================== Página Notion =====================

def notion_create_page(parent_page_id: str,
                       token: str,
                       page_title: str,
                       df: pd.DataFrame,
                       resumen_df: pd.DataFrame,
                       meta: dict,
                       chart_urls: dict,
                       insights: dict,
                       acciones_ai: dict | None = None):
    blocks = []
    blocks.append(_h1(page_title))

    # Resumen Ejecutivo
    blocks.append(_h2("Resumen Ejecutivo"))
    blocks.append(_para(f"📅 Fecha del análisis: {meta.get('fecha','')}"))
    blocks.append(_para(f"📂 Fuente de datos: {meta.get('fuente','')}"))
    blocks.append(_para(f"💬 Conversaciones procesadas: {meta.get('total','')}"))
    blocks.append(_para("Durante el periodo analizado se registraron conversaciones en Intercom, procesadas por IA para identificar patrones, problemas recurrentes y oportunidades de mejora."))

    # KPIs a la vista
    blocks.append(_h2("KPIs a la vista"))
    kpis = {
        "Tickets resueltos": df["tickets_resueltos"].iloc[0] if "tickets_resueltos" in df.columns and len(df) else "—",
        "% 1ra respuesta": df["tasa_1ra_respuesta"].iloc[0] if "tasa_1ra_respuesta" in df.columns and len(df) else "—",
        "Tiempo medio de resolución": df["ttr_horas"].iloc[0] if "ttr_horas" in df.columns and len(df) else "—",
        "Satisfacción (CSAT)": df["csat"].iloc[0] if "csat" in df.columns and len(df) else "—",
    }
    for k, v in kpis.items():
        blocks.append(_bullet(f"{k}: {v}"))

    # Top 3 issues
    top3 = df["issue_group"].value_counts().head(3)
    blocks.append(_h2("Top 3 issues"))
    for issue, casos in top3.items():
        pct = f"{(casos/len(df)*100):.0f}%"
        blocks.append(_bullet(f"{issue} → {casos} casos ({pct})"))
    if insights.get("top_issues"):
        blocks.append(_callout(insights["top_issues"], icon="💡"))

    # Gráficos + insights (URLs válidas únicamente)
    blk = _image_external_if_valid(chart_urls.get("top_issues"), "Top Issues")
    if blk: blocks.append(blk)
    blk = _image_external_if_valid(chart_urls.get("urgencia_pie"), "Distribución de Urgencias")
    if blk: blocks.append(blk)
    if insights.get("urgencia_pie"): blocks.append(_callout(insights["urgencia_pie"], icon="💡"))
    blk = _image_external_if_valid(chart_urls.get("sentimiento_pie"), "Distribución de Sentimientos")
    if blk: blocks.append(blk)
    if insights.get("sentimiento_pie"): blocks.append(_callout(insights["sentimiento_pie"], icon="💡"))
    blk = _image_external_if_valid(chart_urls.get("urgencia_por_issue"), "Urgencia por Issue")
    if blk: blocks.append(blk)
    if insights.get("urgencia_por_issue"): blocks.append(_callout(insights["urgencia_por_issue"], icon="💡"))
    blk = _image_external_if_valid(chart_urls.get("urgencia_top_issues"), "Urgencias en Top Issues (agrupadas)")
    if blk: blocks.append(blk)
    if insights.get("urgencia_top_issues"): blocks.append(_callout(insights["urgencia_top_issues"], icon="💡"))
    blk = _image_external_if_valid(chart_urls.get("canal_por_issue"), "Canal por Issue")
    if blk: blocks.append(blk)
    if insights.get("canal_por_issue"): blocks.append(_callout(insights["canal_por_issue"], icon="💡"))

    # Categorizaciones Manuales
    blocks.append(_h2("Análisis de Categorizaciones Manuales"))
    for k, v in df["tema_norm"].value_counts().head(5).items():
        blocks.append(_bullet(f"Tema • {k}: {v}"))
    for k, v in df["motivo_norm"].value_counts().head(5).items():
        blocks.append(_bullet(f"Motivo • {k}: {v}"))

    # Issues Detallados (tabla nativa con hipervínculos saneados)
    blocks.append(_h2("Issues Detallados"))
    blocks.append(build_issues_table_block(resumen_df))

    # Acciones A Evaluar
    blocks.extend(build_actions_section_blocks(resumen_df, top_n=5, acciones_ai=acciones_ai))

    # --------- SANITIZACIÓN PREVIA A NOTION ---------
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

    # Retry sin NINGÚN link si Notion rechaza por URL
    if not resp.ok and ("Invalid URL" in (resp.text or "") or "link" in (resp.text or "").lower()):
        def strip_all_links(blks: list[dict]) -> list[dict]:
            blks = json.loads(json.dumps(blks))
            for b in blks:
                t = b.get("type")
                if t in ("paragraph","bulleted_list_item","to_do","heading_1","heading_2","heading_3","callout"):
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
                    # dejamos imágenes (ya sanitizadas). Quitamos links en caption si hubiese.
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
        gemini_model: str = "gemini-1.5-flash"):

    os.makedirs(out_dir, exist_ok=True)

    # Lectura + normalización + taxonomías
    df = load_csv_robusto(csv_path)
    df = normalize_columns(df)
    for col in ["tema","motivo","submotivo","urgencia","canal","area","sentimiento","categoria"]:
        if col in df.columns:
            df[col] = df[col].astype(object)
    df = enforce_taxonomy(df)

    # Issue grouping
    df["texto_base"] = df.apply(build_text_base, axis=1)
    df["issue_group"] = df["texto_base"].apply(assign_issue_group)

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
                      "palabras_clave","issue_group","risk","sla_now","taxonomy_flag",
                      "link_a_intercom","id_intercom"]
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

    # Gemini: insights y acciones
    insights = {}
    if gemini_api_key:
        for name, obj in [
            ("top_issues", top_counts.to_dict()),
            ("urgencia_pie", urg_counts.to_dict()),
            ("sentimiento_pie", sent_counts.to_dict()),
            ("urgencia_por_issue", urg_issue_ct),
            ("urgencia_top_issues", urg_top_ct),
            ("canal_por_issue", canal_issue_ct),
        ]:
            txt = ai_insight_for_chart(name, obj, api_key=gemini_api_key, model=gemini_model)
            if txt:
                insights[name] = txt

    acciones_ai = {}
    if gemini_api_key:
        for _, row in resumen_df.head(5).iterrows():
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
            if add:
                acciones_ai[issue] = add

    # Notion
    if notion_token and notion_parent:
        page_title = f"Reporte CX – {date.today().isoformat()}"
        meta = {"fecha": date.today().isoformat(), "fuente": os.path.basename(csv_path), "total": total}
        notion_create_page(
            parent_page_id=notion_parent,
            token=notion_token,
            page_title=page_title,
            df=df,
            resumen_df=resumen_df,
            meta=meta,
            chart_urls=chart_urls,
            insights=insights,
            acciones_ai=acciones_ai if acciones_ai else None
        )
        print("✅ Publicado en Notion.")
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
        gemini_model=args.gemini_model
    )

if __name__ == "__main__":
    main()




