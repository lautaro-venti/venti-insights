# -*- coding: utf-8 -*-
"""
Venti – Insight IA: Análisis semanal de conversaciones Intercom
- Lector CSV robusto (encodings/sep + fallback Excel)
- Normaliza columnas y taxonomías
- Agrupa por issues (heurística)
- Flags (risk, SLA ahora, taxonomy_flag)
- KPIs: urgencia, sentimiento, temas y motivos
- Exporta CSVs + gráficos PNG
- Genera PDF profesional con texto y gráficos embebidos (matplotlib)
- Publica reporte en Notion con bloques nativos

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
        "rol": ["rol","role"]
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

def compare_with_prev(issues_df: pd.DataFrame, hist_dir="./hist") -> pd.DataFrame:
    os.makedirs(hist_dir, exist_ok=True)
    today = date.today().isoformat()
    snap_path = os.path.join(hist_dir, f"issues_{today}.csv")
    issues_df.to_csv(snap_path, index=False, encoding="utf-8")

    prevs = sorted([p for p in os.listdir(hist_dir) if p.startswith("issues_") and p.endswith(".csv")])
    if len(prevs) < 2:
        issues_df["wow_change_pct"] = ""
        issues_df["anomaly_flag"] = False
        return issues_df

    prev_path = os.path.join(hist_dir, prevs[-2])
    prev = pd.read_csv(prev_path)
    prev_map = dict(zip(prev["issue"], prev["casos"]))

    def calc(issue, casos):
        p = prev_map.get(issue, 0)
        if p == 0:
            return 100.0, True if casos >= 10 else False
        delta = (casos - p) / max(p, 1) * 100.0
        return round(delta, 1), (delta >= 50.0 and casos >= 10)

    issues_df["wow_change_pct"], issues_df["anomaly_flag"] = zip(*issues_df.apply(lambda r: calc(r["issue"], r["casos"]), axis=1))
    return issues_df

# ===================== Notion (bloques nativos) =====================

def _rt(text: str):
    s = str(text or "")
    chunks, maxlen = [], 1800
    while s:
        chunks.append({"type": "text", "text": {"content": s[:maxlen]}})
        s = s[maxlen:]
    return chunks or [{"type": "text", "text": {"content": ""}}]

def _para(text: str):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}

def _heading(level: int, text: str):
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rt(text)}}

def _bullet(text: str):
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": _rt(text)}}

def df_to_notion_table(df: pd.DataFrame, max_rows=60, max_cols=9):
    df = df.copy()
    cols = list(df.columns)[:max_cols]
    data = df[cols].astype(str).fillna("")
    rows = [cols] + data.head(max_rows).values.tolist()
    width = len(cols)

    table_block = {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": []
        }
    }
    # Header
    table_block["table"]["children"].append({
        "object": "block",
        "type": "table_row",
        "table_row": {"cells": [[*_rt(c)] for c in rows[0]]}
    })
    # Body
    for row in rows[1:]:
        table_block["table"]["children"].append({
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": [[*_rt(str(c))] for c in row]}
        })
    return [table_block]

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

def notion_create_page(parent_page_id: str, token: str, page_title: str, executive_text: str,
                       resumen_df: pd.DataFrame, extra_bullets=None, assets_base_url: str | None = None):
    if extra_bullets is None:
        extra_bullets = []

    blocks = []
    blocks.append(_heading(1, page_title))

    # Resumen Ejecutivo
    blocks.append(_heading(2, "Resumen Ejecutivo"))
    for line in executive_text.strip().splitlines():
        line = line.strip()
        if not line:
            blocks.append(_para(" "))
        elif line.startswith("- "):
            blocks.append(_bullet(line[2:]))
        else:
            blocks.append(_para(line))

    # Gráficos (si hay base pública)
    if assets_base_url:
        blocks.append(_heading(2, "Gráficos"))
        blocks.append(_image(f"{assets_base_url}/top_issues.png", "Top Issues"))
        blocks.append(_image(f"{assets_base_url}/urgencia_por_issue.png", "Urgencia por Issue"))
        blocks.append(_image(f"{assets_base_url}/canal_por_issue.png", "Canal por Issue"))

    # Tabla resumen issues
    blocks.append(_heading(2, "Issues Detallados"))
    cols_for_table = ["issue","casos","canales_top","areas_top","motivos_top","submotivos_top","wow_change_pct","anomaly_flag","ejemplos_intercom"]
    cols_for_table = [c for c in cols_for_table if c in resumen_df.columns]
    table_blocks = df_to_notion_table(resumen_df[cols_for_table].copy(), max_rows=60, max_cols=len(cols_for_table))
    blocks.extend(table_blocks)

    # Acciones recomendadas
    blocks.append(_heading(2, "Acciones recomendadas"))
    for issue, casos in resumen_df.head(5)[["issue","casos"]].values.tolist():
        blocks.append(_heading(3, f"{issue} ({casos})"))
        acts = actions_for_issue(issue)
        blocks.append(_bullet(f"Producto: {acts['Producto']}"))
        blocks.append(_bullet(f"Tech: {acts['Tech']}"))
        blocks.append(_bullet(f"CX: {acts['CX']}"))

    if extra_bullets:
        blocks.append(_heading(2, "Archivos generados"))
        for label in extra_bullets:
            blocks.append(_bullet(label))

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

def _image_block(external_url: str, caption: str = ""):
    cap = [{"type": "text", "text": {"content": caption}}] if caption else []
    return {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {"url": external_url},
            "caption": cap
        }
    }


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
            # evita cortar palabras
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
        # area para la imagen
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
        # Página 1: Portada + resumen ejecutivo
        lines = [
            f"Fecha del análisis: {meta.get('fecha','')}",
            f"Fuente: {meta.get('fuente','')}",
            f"Conversaciones procesadas: {meta.get('total','')}",
            "",
            "Resumen Ejecutivo:",
        ] + [ln for ln in executive_text.splitlines()]
        _pdf_page_text("Reporte Insight IA – Análisis de Conversaciones Intercom", lines, pdf, footer="Venti · Insight IA")

        # Página 2: Distribución de urgencias / sentimientos
        urg_lines = ["Distribución de Urgencias:"]
        for k, v in stats["urg_counts"].items():
            urg_lines.append(f"- {k}: {v}")
        sent_lines = ["", "Distribución de Sentimientos:"]
        for k, v in stats["sent_counts"].items():
            sent_lines.append(f"- {k}: {v}")
        temas_lines = ["", "Temas Principales:"]
        for k, v in stats["tema_counts"]:
            temas_lines.append(f"- {k}: {v}")
        motivos_lines = ["", "Motivos Principales:"]
        for k, v in stats["motivo_counts"]:
            motivos_lines.append(f"- {k}: {v}")
        _pdf_page_text("Resumen de Distribuciones", urg_lines + sent_lines + temas_lines + motivos_lines, pdf)

        # Página 3: Top issues (gráfico)
        _pdf_page_image("Top Issues", chart_paths["top_issues_png"], pdf, caption="Casos por issue (Top 5)")

        # Página 4: Urgencia por issue (gráfico)
        _pdf_page_image("Urgencia por Issue", chart_paths["urgencia_por_issue_png"], pdf)

        # Página 5: Canal por issue (gráfico)
        _pdf_page_image("Canal por Issue", chart_paths["canal_por_issue_png"], pdf)

        # Página 6+: Issues detallados con acciones (texto)
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

    # 6) Resumen por issue
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

    # 7) Drill-down
    df["issue_asignado"] = df["issue_group"]
    cols_export = ["fecha","canal","rol","area","tema_norm","motivo_norm","submotivo_norm",
                   "categoria","urgencia","sentimiento","resumen_ia","insight_ia",
                   "palabras_clave","issue_asignado","risk","sla_now","taxonomy_flag",
                   "link_a_intercom","id_intercom"]
    existing = [c for c in cols_export if c in df.columns]
    drill_df = df[existing].copy()

    # 8) Stats y gráficos
    total = len(df)
    urg_counts = df["urgencia"].value_counts(dropna=False)
    sent_counts = df["sentimiento"].value_counts(dropna=False)
    tema_counts = df["tema_norm"].value_counts().head(5).items()
    motivo_counts = df["motivo_norm"].value_counts().head(5).items()
    top_issues = df["issue_group"].value_counts().head(5)

    # Gráfico Top issues
    fig, ax = plt.subplots(figsize=(8, 5))
    safe_barh(ax, top_issues, "Top Issues", "Casos")
    top_issues_png = os.path.join(out_dir, "top_issues.png")
    plt.tight_layout(); fig.savefig(top_issues_png); plt.close(fig)

    # Urgencia por issue
    urg_issue = pd.crosstab(df["issue_group"], df["urgencia"])
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    bottom = None
    for col in urg_issue.columns:
        vals = urg_issue[col].values
        x = range(len(urg_issue.index))
        if bottom is None:
            ax2.bar(x, vals, label=str(col)); bottom = vals
        else:
            ax2.bar(x, vals, bottom=bottom, label=str(col)); bottom = [b+v for b, v in zip(bottom, vals)]
    ax2.set_xticks(range(len(urg_issue.index)))
    ax2.set_xticklabels(list(urg_issue.index), rotation=45, ha="right")
    ax2.set_title("Urgencia por Issue"); ax2.set_ylabel("Casos"); ax2.legend()
    urg_issue_png = os.path.join(out_dir, "urgencia_por_issue.png")
    plt.tight_layout(); fig2.savefig(urg_issue_png); plt.close(fig2)

    # Canal por issue
    canal_issue = pd.crosstab(df["issue_group"], df["canal"])
    fig3, ax3 = plt.subplots(figsize=(9, 6))
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

    # 9) Comparativa WoW + anomalías
    resumen_df = compare_with_prev(resumen_df, hist_dir=os.path.join(out_dir, "hist"))

    # 10) Guardar CSVs
    issues_csv = os.path.join(out_dir, "issues_resumen.csv")
    drill_csv = os.path.join(out_dir, "casos_con_issue.csv")
    resumen_df.to_csv(issues_csv, index=False, encoding="utf-8")
    drill_df.to_csv(drill_csv, index=False, encoding="utf-8")
    
    # 11) Resumen Ejecutivo (texto ENRIQUECIDO)
    urg_total = sum(int(v) for v in urg_counts.values)
    urg_fila = [(k, int(v), _pct(int(v), urg_total)) for k, v in urg_counts.items()]
    sent_total = sum(int(v) for v in sent_counts.values)
    sent_fila = [(k, int(v), _pct(int(v), sent_total)) for k, v in sent_counts.items()]

    top3 = df["issue_group"].value_counts().head(3)
    top3_lineas = [f"{i} → {c} casos ({_pct(int(c), total)})" for i, c in top3.items()]

    temas_top = [f"{k} → {v} casos" for k, v in list(tema_counts)]
    motivos_top = [f"{k} → {v} casos" for k, v in list(motivo_counts)]

    executive_text = f"""
    Reporte Insight IA – Análisis de Conversaciones Intercom

    📅 Fecha del análisis: {date.today().isoformat()}
    📂 Fuente de datos: {os.path.basename(csv_path)}
    💬 Conversaciones procesadas: {total}

    1. Resumen Ejecutivo
    Durante el periodo analizado se registraron {total} conversaciones en Intercom, procesadas por IA para identificar patrones, problemas recurrentes y oportunidades de mejora.

    Top 3 issues:
    - {top3_lineas[0] if len(top3_lineas)>0 else '-'}
    - {top3_lineas[1] if len(top3_lineas)>1 else '-'}
    - {top3_lineas[2] if len(top3_lineas)>2 else '-'}

    💡 Insight Global: El { _pct(int(top3.sum()), total) } de los contactos están vinculados a problemas operativos post-compra, por lo que las mejoras deben concentrarse en pagos, accesos y autogestión del usuario.

    2. Distribución de Urgencias
    {os.linesep.join([f"- {k}: {v} ({p})" for k,v,p in urg_fila])}

    3. Distribución de Sentimientos
    {os.linesep.join([f"- {k}: {v} ({p})" for k,v,p in sent_fila])}

    4. Temas y Motivos principales
    Temas:
    {os.linesep.join([f"- {t}" for t in temas_top])}

    Motivos:
    {os.linesep.join([f"- {m}" for m in motivos_top])}

    💡 Insight: Más del 50% se concentra en entrega de entradas y cambios post-compra → foco en flujos de autogestión.

    5. Recomendaciones de Proyecto
    - Feature: Autogestión total de entradas (descarga, reenvío y transferencia) desde app/web.
    - Feature: Verificación/edición de datos de usuario antes de compra.
    - Automatización: Respuestas automáticas con reenvío cuando se detecta “No recibí mi entrada”.
    - Soporte proactivo: Alertas internas para urgencia alta + notificaciones push al usuario.
    """.strip()
    
    # 12) Generar PDF con gráficos embebidos
    pdf_path = create_pdf_report(
        out_dir=out_dir,
        meta={"fecha": date.today().isoformat(), "fuente": os.path.basename(csv_path), "total": total},
        executive_text=executive_text,
        stats={
            "urg_counts": urg_counts.to_dict(),
            "sent_counts": sent_counts.to_dict(),
            "tema_counts": list(tema_counts),
            "motivo_counts": list(motivo_counts),
        },
        chart_paths={
            "top_issues_png": top_issues_png,
            "urgencia_por_issue_png": urg_issue_png,
            "canal_por_issue_png": canal_issue_png,
        },
        resumen_df=resumen_df
    )

    # 13) Markdown (por si lo querés también)
    md_path = os.path.join(out_dir, "venti_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Insight IA – Análisis semanal de conversaciones Intercom\n\n")
        f.write(executive_text + "\n\n")
        f.write("## Resumen por issue\n\n")
        try:
            f.write(resumen_df.to_markdown(index=False))
        except Exception:
            f.write(resumen_df.to_csv(index=False))
        f.write("\n\n---\n")
        f.write("> Archivos locales generados: issues_resumen.csv, casos_con_issue.csv, top_issues.png, urgencia_por_issue.png, canal_por_issue.png\n")

    return {
        "issues_csv": issues_csv,
        "drill_csv": drill_csv,
        "report_md": md_path,
        "top_issues_png": top_issues_png,
        "urgencia_por_issue_png": urg_issue_png,
        "canal_por_issue_png": canal_issue_png,
        "pdf_report": pdf_path,
        "resumen_df": resumen_df,
        "executive_text": executive_text,
        "total": total
    }

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Insight IA – Análisis semanal de conversaciones Intercom (Venti)")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de conversaciones")
    ap.add_argument("--out", default="./salida", help="Directorio de salida (por defecto ./salida)")
    ap.add_argument("--notion_token", default=os.getenv("NOTION_TOKEN"), help="Token de Notion (o env NOTION_TOKEN)")
    ap.add_argument("--notion_parent", default=os.getenv("NOTION_PARENT_PAGE_ID"), help="ID de página padre en Notion (o env NOTION_PARENT_PAGE_ID)")
    ap.add_argument("--publish_github", action="store_true",
                help="Si se pasa, publica PNGs en el repo y usa su URL raw en Notion")
    ap.add_argument("--github_repo_path", default=None,
                help="Ruta local al repo clonado (ej: A:\\Venti CX\\venti-insights)")
    ap.add_argument("--github_branch", default="main",
                help="Branch a usar para el push (default: main)")
    ap.add_argument("--assets_base_url", default=None,
                help="Base URL pública para imágenes (si ya están hosteadas)")
    args = ap.parse_args()

    print("▶ Script iniciado")
    print("CSV:", args.csv)
    print("OUT:", args.out)
    print("NOTION_TOKEN:", "OK" if args.notion_token else "FALTA")
    print("PARENT:", args.notion_parent)

    paths = run_weekly_report(args.csv, args.out)

    # Publicar en Notion si hay credenciales
    assets_base_url = args.assets_base_url

# Publicación automática en GitHub si se pidió
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
            f"Gráfico Top Issues: {paths['top_issues_png']}",
            f"Gráfico Urgencia por Issue: {paths['urgencia_por_issue_png']}",
            f"Gráfico Canal por Issue: {paths['canal_por_issue_png']}",
            f"PDF: {paths['pdf_report']}",
        ]
        notion_create_page(
            parent_page_id=args.notion_parent,
            token=args.notion_token,
            page_title=title,
            executive_text=paths["executive_text"],
            resumen_df=paths["resumen_df"],
            extra_bullets=extra,
            assets_base_url=assets_base_url,   # <--- acá va la URL pública
        )
        print("✅ Publicado en Notion.")
    else:
        print("ℹ️ No se detectó NOTION_TOKEN / NOTION_PARENT_PAGE_ID. Se generaron los archivos locales.")
        print(json.dumps({
            "issues_csv": paths["issues_csv"],
            "drill_csv": paths["drill_csv"],
            "report_md": paths["report_md"],
            "pdf_report": paths["pdf_report"]
        }, indent=2, ensure_ascii=False))
        
# --- NUEVO: imports para publicar en GitHub ---

def _run_git(cmd, cwd):
    """Ejecuta un comando git y retorna salida (str). Lanza excepción si falla."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return (p.stdout or "").strip()

def _parse_remote_origin(remote_url: str):
    """
    Devuelve (owner, repo) a partir de:
      - https://github.com/owner/repo.git
      - git@github.com:owner/repo.git
    """
    u = remote_url.strip()
    if u.endswith(".git"):
        u = u[:-4]
    if "github.com/" in u:
        # https
        owner_repo = u.split("github.com/")[-1]
    elif "github.com:" in u:
        # ssh
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
    """
    Copia PNGs de out_dir -> <repo>/reports/<YYYY-MM-DD>/,
    hace git add/commit/push y retorna la base RAW pública.
    """
    if files is None:
        files = ["top_issues.png", "urgencia_por_issue.png", "canal_por_issue.png"]
    if date_subdir is None:
        date_subdir = date.today().isoformat()

    # 1) sanity
    if not os.path.isdir(repo_path):
        raise RuntimeError(f"Repo path no existe: {repo_path}")
    for fn in files:
        p = os.path.join(out_dir, fn)
        if not os.path.exists(p):
            raise RuntimeError(f"No existe imagen: {p}")

    # 2) destino y copia
    dest_dir = os.path.join(repo_path, "reports", date_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    for fn in files:
        shutil.copy2(os.path.join(out_dir, fn), os.path.join(dest_dir, fn))

    # 3) git add/commit/push
    _run_git(["git", "add", "."], cwd=repo_path)
    try:
        _run_git(["git", "commit", "-m", f"Report {date_subdir}: PNG charts"], cwd=repo_path)
    except subprocess.CalledProcessError as e:
        # Sin cambios para commitear es válido
        if "nothing to commit" not in (e.stderr or "") and "nothing to commit" not in (e.stdout or ""):
            raise

    _run_git(["git", "push", "origin", branch], cwd=repo_path)

    # 4) Construir base RAW
    remote = _run_git(["git", "config", "--get", "remote.origin.url"], cwd=repo_path)
    owner, repo = _parse_remote_origin(remote)
    base_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/reports/{date_subdir}"
    return base_raw

# --- NUEVO: bloque imagen para Notion ---
def _image(url: str, caption: str | None = None):
    b = {
        "object": "block",
        "type": "image",
        "image": {"type": "external", "external": {"url": url}}
    }
    if caption:
        b["image"]["caption"] = [{"type": "text", "text": {"content": caption}}]
    return b

if __name__ == "__main__":
    main()

