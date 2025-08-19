# -*- coding: utf-8 -*-
"""
Venti – Insight IA: Análisis semanal de conversaciones Intercom
- Lector CSV robusto (encodings/sep + fallback Excel)
- Normaliza columnas y taxonomías
- Agrupa por issues (heurística)
- Flags (risk, SLA ahora, taxonomy_flag)
- KPIs placeholders: tickets resueltos, FTR, TTR, CSAT (si no vienen en CSV queda "—")
- Exporta CSVs + gráficos PNG (Top Issues, Urgencia vs Issue, Canal por Issue, Urgencia (pie), Sentimiento (pie))
- Genera PDF profesional con texto y gráficos embebidos (matplotlib)
- Publica reporte en Notion con narrativa (sin tablas crudas), imágenes con caption e insights, e hipervínculos "Ejemplo N" a Intercom
- Opcional: publica PNGs en GitHub y usa URLs públicas en Notion

Requisitos:
  pip install pandas matplotlib requests
"""

import os
import re
import json
import argparse
from datetime import date
import requests
import pandas as pd
import subprocess
import shutil

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ===================== Taxonomías cerradas (ajustables) =====================

VALID_TEMAS = set([
    "eventos - user ticket",
    "eventos - user productora",
    "lead comercial",
    "anuncios & notificaciones",
    "duplicado",
    "desvío a intercom",
    "sin respuesta",
])

VALID_MOTIVOS = set([
    "caso excepcional",
    "reenvío",
    "estafa por reventa",
    "compra externa a venti",
    "consulta por evento",
    "team leads & públicas",
    "devolución",
    "pagos",
    "seguridad",
    "evento reprogramado",
    "evento cancelado",
    "contacto comercial",
    "anuncios & notificaciones",
    "duplicado",
    "desvío a intercom",
    "no recibí mi entrada",
    "sdu (sist. de usuarios)",
    "transferencia de entradas",
    "qr shield",
    "venti swap",
    "reporte",
    "carga masiva",
    "envío de invitaciones",
    "carga de un evento",
    "servicios operativos",
    "solicitud de reembolso",
    "adelantos",
    "liquidaciones",
    "estado de cuenta",
    "datos de cuenta",
    "altas en venti",
    "app de validación",
    "validadores",
    "organización de accesos en el evento",
    "facturación",
    "sin respuesta",
    "reclamo de usuario",
    "consulta sobre uso de la plataforma",
    "desvinculación de personal",
])

VALID_SUBMOTIVOS = set([])  # opcional

# ===================== Utilidades base =====================

def load_csv_robusto(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1", "utf-16", "utf-16-le", "utf-16-be"]
    seps = [",", ";", "\t", None]  # None => autodetect con engine="python"
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
    # fallback Excel renombrado
    try:
        df = pd.read_excel(csv_path, dtype=str)
        return df
    except Exception:
        pass
    raise RuntimeError(f"No pude leer el CSV con encodings/sep comunes. Último error: {last_err}")

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
        # KPIs si vinieran desde Zapier
        "tickets_resueltos": ["tickets_resueltos","tickets","resolved"],
        "ftr_rate": ["ftr_rate","first_time_resolution","first_response_rate"],
        "ttr_horas": ["ttr_horas","time_to_resolve_hours","resolution_time_h"],
        "csat": ["csat","satisfaccion","satisfaction"],
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

    # columnas mínimas
    for canon in ["resumen_ia","insight_ia","palabras_clave","canal","area","tema",
                  "motivo","submotivo","urgencia","sentimiento","categoria",
                  "link_a_intercom","id_intercom","fecha","rol"]:
        if canon not in df.columns:
            df[canon] = ""

    # KPIs placeholders si no vienen en CSV
    for kpi in ["tickets_resueltos","ftr_rate","ttr_horas","csat"]:
        if kpi not in df.columns:
            df[kpi] = "—"

    return df

def map_to_catalog(value, catalog):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "", False
        v = str(value).strip().lower()
    except Exception:
        v = ""
    if not v or v in ("nan", "none"):
        return "", False
    if v in catalog:
        return v, True
    for item in catalog:
        if v in item or item in v:
            return item, True
    return v, False

def _pct(n, d):
    try:
        return f"{(n/d*100):.0f}%"
    except Exception:
        return "0%"

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
    txt = " ".join(str(row.get(c, "") or "") for c in
                   ["resumen_ia","insight_ia","palabras_clave","tema_norm","motivo_norm","submotivo_norm"]).lower()
    urg = _safe_lower(row.get("urgencia", ""))
    canal = _safe_lower(row.get("canal", ""))

    risk = "LOW"
    if any(k in txt for k in ["estafa","fraude","no puedo entrar","no puedo ingresar","rechazad"]):
        risk = "HIGH"
    elif urg in ("alta","high"):
        risk = "HIGH"
    elif urg in ("media","medium"):
        risk = "MEDIUM"

    sla_now = (canal in ("whatsapp","instagram")) and (risk == "HIGH")
    return pd.Series({"risk": risk, "sla_now": sla_now})

def safe_barh(ax, counts, title, xlabel):
    labels = list(counts.index)
    values = list(counts.values)
    y_pos = range(len(labels))
    ax.barh(y_pos, values)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

# ===================== Histórico semana/semana =====================

def compare_with_prev(issues_df: pd.DataFrame, hist_dir="./hist") -> pd.DataFrame:
    os.makedirs(hist_dir, exist_ok=True)
    today = date.today().isoformat()
    snap_path = os.path.join(hist_dir, f"issues_{today}.csv")
    issues_df.to_csv(snap_path, index=False, encoding="utf-8")

    prevs = sorted([p for p in os.listdir(hist_dir) if p.startswith("issues_") and p.endswith(".csv")])
    if len(prevs) < 2:
        issues_df["wow_change_pct"] = 0.0
        issues_df["anomaly_flag"] = False
        return issues_df

    prev_path = os.path.join(hist_dir, prevs[-2])
    prev = pd.read_csv(prev_path)
    prev_map = dict(zip(prev.get("issue", []), prev.get("casos", [])))

    def calc(issue, casos):
        p = prev_map.get(issue, 0)
        try:
            if p == 0:
                return 100.0, True if casos >= 10 else False
            delta = (casos - p) / max(p, 1) * 100.0
            return round(delta, 1), (delta >= 50.0 and casos >= 10)
        except Exception:
            return 0.0, False

    issues_df["wow_change_pct"], issues_df["anomaly_flag"] = zip(*issues_df.apply(lambda r: calc(r["issue"], r["casos"]), axis=1))
    return issues_df

# ===================== Notion (bloques nativos, narrativa) =====================

def _text(content: str, href: str | None = None):
    obj = {"type": "text", "text": {"content": str(content)}}
    if href:
        obj["text"]["link"] = {"url": href}
    return obj

def _rt(*spans):
    # spans: list of rich_text objects already constructed
    if not spans:
        return [{"type": "text", "text": {"content": ""}}]
    return list(spans)

def _para(rich_spans):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_spans}}

def _heading(level: int, text: str):
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rt(_text(text))}}

def _bullet(rich_spans):
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich_spans}}

def _image(url: str, caption: str | None = None):
    b = {"object": "block", "type": "image", "image": {"type": "external", "external": {"url": url}}}
    if caption:
        b["image"]["caption"] = _rt(_text(caption))
    return b

ACTION_LIBRARY = {
    "Pagos / cobros": {
        "Producto": "- Mostrar causa de rechazo y reintentos guiados.\n- Guardar tarjeta segura (1-click).",
        "Tech": "- Logs de PSP + alertas por BIN/issuer.\n- Retries con backoff en fallas de red.",
        "CX": "- Macro paso a paso por medio de pago.\n- Comunicar reserva temporal."
    },
    "Entrega de entradas": {
        "Producto": "- CTA visible para reenvío + confirmación en UI.",
        "Tech": "- Job de reenvío idempotente.\n- Monitoreo de bounce/spam.",
        "CX": "- Bot de autogestión: reenvío por mail/WhatsApp."
    },
    "QR / Validación en acceso": {
        "Producto": "- Feedback claro de estado del QR (válido/usado/bloqueado).",
        "Tech": "- Telemetría de validadores + health checks.",
        "CX": "- Guía de acceso y resolución de errores comunes."
    },
    "Transferencia / titularidad": {
        "Producto": "- Flujo guiado de cambio de titularidad con confirmación.",
        "Tech": "- Auditoría/registro de transferencias.",
        "CX": "- Macro con costos/plazos y límites."
    },
    "Reembolso / devolución": {
        "Producto": "- Estado visible del reembolso y tiempos estimados.",
        "Tech": "- Idempotencia + conciliación con PSP.",
        "CX": "- Macro de seguimiento y expectativas."
    },
}

def actions_for_issue(issue: str):
    d = ACTION_LIBRARY.get(issue, {})
    if not d:
        d = {
            "Producto": "- Quick wins de UX para reducir fricción.",
            "Tech": "- Registrar error types y agregar tracing/dashboards.",
            "CX": "- Macro de contención + FAQ específica."
        }
    return d

# ===== Insights automáticos para captions debajo de gráficos =====

def insight_urgencia(urg_counts: pd.Series) -> str:
    if urg_counts.empty:
        return "Sin datos de urgencia."
    total = int(urg_counts.sum())
    mayor = urg_counts.sort_values(ascending=False).index.tolist()[0]
    pct = _pct(int(urg_counts[mayor]), total)
    return f"La mayor proporción de casos es de urgencia **{mayor}** ({pct}). Priorizar playbooks y SLAs para este nivel."

def insight_sentimiento(sent_counts: pd.Series) -> str:
    if sent_counts.empty:
        return "Sin datos de sentimiento."
    total = int(sent_counts.sum())
    mayor = sent_counts.sort_values(ascending=False).index.tolist()[0]
    pct = _pct(int(sent_counts[mayor]), total)
    return f"Predomina el sentimiento **{mayor}** ({pct}). Reforzar mensajes proactivos y personalización para elevar CSAT."

def insight_top_issues(top_issues: pd.Series, total: int) -> str:
    if top_issues.empty:
        return "Sin issues destacados."
    top3_sum = int(top_issues.head(3).sum())
    return f"Los 3 principales issues concentran el {_pct(top3_sum, total)} del volumen. Enfocar mejoras en estos frentes."

# ===================== PDF con matplotlib =====================

def _pdf_page_text(title, lines, pdf, footer=None):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 vertical en pulgadas
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    y = 0.95
    ax.text(0.05, y, title, fontsize=18, weight="bold", va="top")
    y -= 0.04
    for ln in lines:
        if not ln:
            y -= 0.02
            continue
        # Split largo en múltiples líneas
        chunks = []
        text = str(ln)
        while len(text) > 120:
            cut = text[:120]
            space = cut.rfind(" ")
            if space < 60: space = 120
            chunks.append(text[:space])
            text = text[space:].lstrip()
        chunks.append(text)
        for c in chunks:
            ax.text(0.05, y, c, fontsize=11, va="top")
            y -= 0.026
        y -= 0.004
        if y < 0.08:
            break
    if footer:
        ax.text(0.5, 0.03, footer, fontsize=8, ha="center", va="bottom", alpha=0.6)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def _pdf_page_image(title, img_path, pdf, caption=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    ax.text(0.05, 0.95, title, fontsize=18, weight="bold", va="top")
    if os.path.exists(img_path):
        image = plt.imread(img_path)
        ax_img = fig.add_axes([0.05, 0.15, 0.9, 0.72])
        ax_img.axis("off")
        ax_img.imshow(image)
        if caption:
            ax.text(0.05, 0.10, caption, fontsize=10, va="top", alpha=0.8)
    else:
        ax.text(0.05, 0.85, f"(No se encontró la imagen: {img_path})", fontsize=12, color="red")
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def create_pdf_report(out_dir: str, meta: dict, executive_text: str, stats: dict, chart_paths: dict, resumen_df: pd.DataFrame) -> str:
    pdf_path = os.path.join(out_dir, f"Reporte_InsightIA_{date.today().isoformat()}.pdf")
    with PdfPages(pdf_path) as pdf:
        lines = [
            f"Fecha del análisis: {meta.get('fecha','')}",
            f"Fuente: {meta.get('fuente','')}",
            f"Conversaciones procesadas: {meta.get('total','')}",
            "",
            "Resumen Ejecutivo:",
        ] + [ln for ln in executive_text.splitlines()]
        _pdf_page_text("Reporte Insight IA – Análisis de Conversaciones Intercom", lines, pdf, footer="Venti · Insight IA")

        # Gráficos
        _pdf_page_image("Top Issues", chart_paths["top_issues_png"], pdf, caption=insight_top_issues(stats["top_issues_series"], meta.get("total", 0)))
        _pdf_page_image("Urgencia por Issue", chart_paths["urgencia_por_issue_png"], pdf, caption=insight_urgencia(pd.Series(stats["urg_counts"])) )
        _pdf_page_image("Canal por Issue", chart_paths["canal_por_issue_png"], pdf)
        _pdf_page_image("Distribución de Urgencias", chart_paths["urgencia_pie_png"], pdf, caption=insight_urgencia(pd.Series(stats["urg_counts"])) )
        _pdf_page_image("Distribución de Sentimientos", chart_paths["sentimiento_pie_png"], pdf, caption=insight_sentimiento(pd.Series(stats["sent_counts"])) )

        # Acciones
        lines = ["Issues Detallados (Top 5):"]
        for issue, casos in resumen_df.head(5)[["issue","casos"]].values.tolist():
            lines.append(f"• {issue} → {casos} casos")
            acts = actions_for_issue(issue)
            lines.append(f"  - Producto: {acts['Producto']}")
            lines.append(f"  - Tech: {acts['Tech']}")
            lines.append(f"  - CX: {acts['CX']}")
            lines.append("")
        _pdf_page_text("Prioridades y Acciones Recomendadas", lines, pdf)

    return pdf_path

# ===================== Publicación GitHub (opcional) =====================

def _run_git(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
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
        files = ["top_issues.png", "urgencia_por_issue.png", "canal_por_issue.png", "urgencia_pie.png", "sentimiento_pie.png"]
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
        _run_git(["git", "commit", "-m", f"Report {date_subdir}: assets"], cwd=repo_path)
    except subprocess.CalledProcessError as e:
        if "nothing to commit" not in (e.stderr or "") and "nothing to commit" not in (e.stdout or ""):
            raise

    _run_git(["git", "push", "origin", branch], cwd=repo_path)

    remote = _run_git(["git", "config", "--get", "remote.origin.url"], cwd=repo_path)
    owner, repo = _parse_remote_origin(remote)
    base_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/reports/{date_subdir}"
    return base_raw

# ===================== Core del reporte =====================

def run_weekly_report(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Lectura CSV robusta
    df = load_csv_robusto(csv_path)

    # 2) Normalización
    df = normalize_columns(df)
    for col in ["tema","motivo","submotivo","urgencia","canal","area","sentimiento","categoria"]:
        if col in df.columns:
            df[col] = df[col].astype(object)

    # 3) Taxonomías
    df = enforce_taxonomy(df)

    # 4) Issue grouping
    df["texto_base"] = df.apply(build_text_base, axis=1)
    df["issue_group"] = df["texto_base"].apply(assign_issue_group)

    # 5) Flags
    flags = df.apply(compute_flags, axis=1)
    for c in flags.columns:
        df[c] = flags[c]

    # 6) Resumen por issue (para narrativa + ejemplos linkeados)
    rows = []
    for issue, grp in df.groupby("issue_group"):
        canales = top_values(grp["canal"]) 
        areas = top_values(grp["area"]) 
        motivos = top_values(grp["motivo_norm"]) 
        submotivos = top_values(grp["submotivo_norm"]) 
        links = grp.loc[grp["link_a_intercom"].astype(str).str.len() > 0, "link_a_intercom"].head(3).astype(str).tolist()
        rows.append({
            "issue": issue,
            "casos": int(len(grp)),
            "canales_top": canales,
            "areas_top": areas,
            "motivos_top": motivos,
            "submotivos_top": submotivos,
            "ejemplos_links": links,
        })
    resumen_df = pd.DataFrame(rows).sort_values("casos", ascending=False)

    # 7) Drill-down (se mantiene para CSV)
    df["issue_asignado"] = df["issue_group"]
    cols_export = ["fecha","canal","rol","area","tema_norm","motivo_norm","submotivo_norm",
                   "categoria","urgencia","sentimiento","resumen_ia","insight_ia",
                   "palabras_clave","issue_asignado","risk","sla_now","taxonomy_flag",
                   "link_a_intercom","id_intercom"]
    existing = [c for c in cols_export if c in df.columns]
    drill_df = df[existing].copy()

    # 8) Stats & gráficos
    total = len(df)
    urg_counts = df["urgencia"].fillna("").replace("nan", "").value_counts(dropna=False)
    sent_counts = df["sentimiento"].fillna("").replace("nan", "").value_counts(dropna=False)
    top_issues = df["issue_group"].value_counts().head(5)

    # Top issues (barh)
    fig, ax = plt.subplots(figsize=(8, 5))
    safe_barh(ax, top_issues, "Top Issues", "Casos")
    top_issues_png = os.path.join(out_dir, "top_issues.png")
    plt.tight_layout(); fig.savefig(top_issues_png); plt.close(fig)

    # Urgencia vs Issue (grouped bars)
    urg_issue = pd.crosstab(df["issue_group"], df["urgencia"]).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = range(len(urg_issue.index))
    width = 0.18
    for i, col in enumerate(urg_issue.columns):
        ax2.bar([xi + i*width for xi in x], urg_issue[col].values, width=width, label=str(col))
    ax2.set_xticks([xi + (len(urg_issue.columns)-1)*width/2 for xi in x])
    ax2.set_xticklabels(list(urg_issue.index), rotation=45, ha="right")
    ax2.set_title("Distribución de Urgencias en Top Issues"); ax2.set_ylabel("Casos"); ax2.legend()
    urg_issue_png = os.path.join(out_dir, "urgencia_por_issue.png")
    plt.tight_layout(); fig2.savefig(urg_issue_png); plt.close(fig2)

    # Canal por issue (stacked bars)
    canal_issue = pd.crosstab(df["issue_group"], df["canal"]).fillna(0)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bottom = None
    for col in canal_issue.columns:
        vals = canal_issue[col].values
        x = range(len(canal_issue.index))
        if bottom is None:
            ax3.bar(x, vals, label=str(col)); bottom = vals
        else:
            ax3.bar(x, vals, bottom=bottom, label=str(col)); bottom = [b+v for b, v in zip(bottom, vals)]
    ax3.set_xticks(range(len(canal_issue.index)))
    ax3.set_xticklabels(list(canal_issue.index), rotation=45, ha="right")
    ax3.set_title("Canal por Issue"); ax3.set_ylabel("Casos"); ax3.legend()
    canal_issue_png = os.path.join(out_dir, "canal_por_issue.png")
    plt.tight_layout(); fig3.savefig(canal_issue_png); plt.close(fig3)

    # Distribución de Urgencias (pie)
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    urg_counts.plot(kind="pie", autopct='%1.0f%%', ax=ax4)
    ax4.set_ylabel("")
    ax4.set_title("Distribución de Urgencias")
    urgencia_pie_png = os.path.join(out_dir, "urgencia_pie.png")
    plt.tight_layout(); fig4.savefig(urgencia_pie_png); plt.close(fig4)

    # Distribución de Sentimientos (pie)
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    sent_counts.plot(kind="pie", autopct='%1.0f%%', ax=ax5)
    ax5.set_ylabel("")
    ax5.set_title("Distribución de Sentimientos")
    sentimiento_pie_png = os.path.join(out_dir, "sentimiento_pie.png")
    plt.tight_layout(); fig5.savefig(sentimiento_pie_png); plt.close(fig5)

    # 9) Comparativa WoW + anomalías
    resumen_df = compare_with_prev(resumen_df, hist_dir=os.path.join(out_dir, "hist"))

    # 10) Guardar CSVs
    issues_csv = os.path.join(out_dir, "issues_resumen.csv")
    drill_csv = os.path.join(out_dir, "casos_con_issue.csv")
    resumen_df.to_csv(issues_csv, index=False, encoding="utf-8")
    drill_df.to_csv(drill_csv, index=False, encoding="utf-8")

    # 11) Narrativa: Resumen Ejecutivo + KPIs placeholder
    top3 = df["issue_group"].value_counts().head(3)
    top3_lineas = [f"{i} → {c} casos ({_pct(int(c), total)})" for i, c in top3.items()]

    executive_text = f"""
Durante el periodo analizado se registraron {total} conversaciones en Intercom, procesadas por IA para identificar patrones, problemas recurrentes y oportunidades de mejora.

📅 Fecha del análisis: {date.today().isoformat()}
📂 Fuente de datos: {os.path.basename(csv_path)}
💬 Conversaciones procesadas: {total}

1. Top 3 issues:
- {top3_lineas[0] if len(top3_lineas)>0 else '-'}
- {top3_lineas[1] if len(top3_lineas)>1 else '-'}
- {top3_lineas[2] if len(top3_lineas)>2 else '-'}

💡 Insight Global: {insight_top_issues(top3, total)}
""".strip()

    # KPIs a la vista (placeholders o valores si vienen)
    kpis = {
        "Tickets resueltos": df.get("tickets_resueltos").iloc[0] if "tickets_resueltos" in df.columns else "—",
        "% First-Time Resolution": df.get("ftr_rate").iloc[0] if "ftr_rate" in df.columns else "—",
        "Tiempo medio de resolución (h)": df.get("ttr_horas").iloc[0] if "ttr_horas" in df.columns else "—",
        "CSAT": df.get("csat").iloc[0] if "csat" in df.columns else "—",
    }

    return {
        "issues_csv": issues_csv,
        "drill_csv": drill_csv,
        "top_issues_png": top_issues_png,
        "urgencia_por_issue_png": urg_issue_png,
        "canal_por_issue_png": canal_issue_png,
        "urgencia_pie_png": urgencia_pie_png,
        "sentimiento_pie_png": sentimiento_pie_png,
        "resumen_df": resumen_df,
        "executive_text": executive_text,
        "kpis": kpis,
        "stats": {
            "urg_counts": urg_counts.to_dict(),
            "sent_counts": sent_counts.to_dict(),
            "top_issues_series": top_issues,
        },
        "total": total,
    }

# ===================== Notion: narrativa + imágenes + links =====================

def notion_create_page(parent_page_id: str, token: str, page_title: str, executive_text: str,
                       resumen_df: pd.DataFrame, imgs: dict, stats: dict, kpis: dict,
                       extra_bullets=None, assets_base_url: str | None = None):
    if extra_bullets is None:
        extra_bullets = []

    blocks = []
    blocks.append(_heading(1, page_title))

    # Resumen Ejecutivo
    blocks.append(_heading(2, "Resumen Ejecutivo"))
    for line in executive_text.splitlines():
        if line.strip():
            blocks.append(_para(_rt(_text(line))))
        else:
            blocks.append(_para(_rt(_text(" "))))

    # KPIs a la vista
    blocks.append(_heading(2, "KPIs a la vista"))
    for k, v in kpis.items():
        blocks.append(_bullet(_rt(_text(f"{k}: {v}"))))

    # Gráficos + insights
    if assets_base_url:
        blocks.append(_heading(2, "Top Issues"))
        blocks.append(_image(f"{assets_base_url}/top_issues.png", caption=insight_top_issues(stats["top_issues_series"], int(stats["top_issues_series"].sum()))))

        blocks.append(_heading(2, "Distribución de Urgencias en Top Issues"))
        blocks.append(_image(f"{assets_base_url}/urgencia_por_issue.png", caption=insight_urgencia(pd.Series(stats["urg_counts"]))))

        blocks.append(_heading(2, "Canal por Issue"))
        blocks.append(_image(f"{assets_base_url}/canal_por_issue.png", caption="Mix de canales por issue. Consolidar autoservicio en los canales de mayor volumen."))

        blocks.append(_heading(2, "Distribución de Urgencias"))
        blocks.append(_image(f"{assets_base_url}/urgencia_pie.png", caption=insight_urgencia(pd.Series(stats["urg_counts"]))))

        blocks.append(_heading(2, "Distribución de Sentimientos"))
        blocks.append(_image(f"{assets_base_url}/sentimiento_pie.png", caption=insight_sentimiento(pd.Series(stats["sent_counts"])) ))

    # Issues Detallados (narrativo, con hipervínculos)
    blocks.append(_heading(2, "Issues Detallados"))
    for _, r in resumen_df.sort_values("casos", ascending=False).iterrows():
        blocks.append(_heading(3, f"{r['issue']} ({r['casos']})"))
        blocks.append(_bullet(_rt(_text(f"Canales más frecuentes: {r['canales_top']}"))))
        blocks.append(_bullet(_rt(_text(f"Áreas más afectadas: {r['areas_top']}"))))
        blocks.append(_bullet(_rt(_text(f"Motivos top: {r['motivos_top']}"))))
        if str(r.get('submotivos_top', '')).strip():
            blocks.append(_bullet(_rt(_text(f"Submotivos top: {r['submotivos_top']}"))))
        # Ejemplos linkeados
        links = r.get("ejemplos_links", []) or []
        if links:
            blocks.append(_para(_rt(_text("Ejemplos:"))))
            for i, url in enumerate(links, start=1):
                blocks.append(_bullet(_rt(_text(f"Ejemplo {i}", href=url))))

    # Acciones recomendadas
    blocks.append(_heading(2, "Acciones a evaluar"))
    for issue, casos in resumen_df.head(5)[["issue","casos"]].values.tolist():
        blocks.append(_heading(3, f"{issue} ({casos})"))
        acts = actions_for_issue(issue)
        for area, txt in acts.items():
            # checklist estilo tareas
            for line in str(txt).split("\n"):
                line = line.strip("- ")
                blocks.append(_bullet(_rt(_text(f"[ ] {area}: {line}"))))

    if extra_bullets:
        blocks.append(_heading(2, "Archivos generados"))
        for label in extra_bullets:
            blocks.append(_bullet(_rt(_text(label))))

    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "properties": {"title": {"title": [{"text": {"content": page_title}}]}},
        "children": blocks
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
    if not resp.ok:
        raise RuntimeError(f"Notion error {resp.status_code}: {resp.text}")
    return resp.json()



# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Insight IA – Análisis semanal de conversaciones Intercom (Venti)")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de conversaciones")
    ap.add_argument("--out", default="./salida", help="Directorio de salida (por defecto ./salida)")
    ap.add_argument("--notion_token", default=os.getenv("NOTION_TOKEN"), help="Token de Notion (o env NOTION_TOKEN)")
    ap.add_argument("--notion_parent", default=os.getenv("NOTION_PARENT_PAGE_ID"), help="ID de página padre en Notion (o env NOTION_PARENT_PAGE_ID)")
    ap.add_argument("--publish_github", action="store_true", help="Si se pasa, publica PNGs en el repo y usa su URL raw en Notion")
    ap.add_argument("--github_repo_path", default=None, help="Ruta local al repo clonado (ej: A:\\Venti CX\\venti-insights)")
    ap.add_argument("--github_branch", default="main", help="Branch a usar para el push (default: main)")
    ap.add_argument("--assets_base_url", default=None, help="Base URL pública para imágenes (si ya están hosteadas)")
    args = ap.parse_args()

    print("▶ Script iniciado")
    print("CSV:", args.csv)
    print("OUT:", args.out)
    print("NOTION_TOKEN:", "OK" if args.notion_token else "FALTA")
    print("PARENT:", args.notion_parent)

    paths = run_weekly_report(args.csv, args.out)

    # Publicación en GitHub si se pidió
    assets_base_url = args.assets_base_url
    if args.publish_github and args.github_repo_path:
        try:
            assets_base_url = publish_images_to_github(
                out_dir=os.path.dirname(paths["top_issues_png"]),
                repo_path=args.github_repo_path,
                branch=args.github_branch,
                date_subdir=date.today().isoformat(),
                files=[
                    os.path.basename(paths["top_issues_png"]),
                    os.path.basename(paths["urgencia_por_issue_png"]),
                    os.path.basename(paths["canal_por_issue_png"]),
                    os.path.basename(paths["urgencia_pie_png"]),
                    os.path.basename(paths["sentimiento_pie_png"]),
                ],
            )
            print(f"🌐 Assets publicados en: {assets_base_url}")
        except Exception as e:
            print(f"⚠️ No pude publicar en GitHub: {e}")

    # Publicar en Notion
    if args.notion_token and args.notion_parent:
        title = f"Reporte CX – {date.today().isoformat()}"
        extra = [
            f"Issues CSV: {paths['issues_csv']}",
            f"Drill-down CSV: {paths['drill_csv']}",
        ]
        notion_create_page(
            parent_page_id=args.notion_parent,
            token=args.notion_token,
            page_title=title,
            executive_text=paths["executive_text"],
            resumen_df=paths["resumen_df"],
            imgs={
                "top_issues_png": paths["top_issues_png"],
                "urgencia_por_issue_png": paths["urgencia_por_issue_png"],
                "canal_por_issue_png": paths["canal_por_issue_png"],
                "urgencia_pie_png": paths["urgencia_pie_png"],
                "sentimiento_pie_png": paths["sentimiento_pie_png"],
            },
            stats=paths["stats"],
            kpis=paths["kpis"],
            extra_bullets=extra,
            assets_base_url=assets_base_url,
        )
        print("✅ Publicado en Notion.")
    else:
        print("ℹ️ No se detectó NOTION_TOKEN / NOTION_PARENT_PAGE_ID. Se generaron los archivos locales.")
        print(json.dumps({
            "issues_csv": paths["issues_csv"],
            "drill_csv": paths["drill_csv"],
        }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
