# -*- coding: utf-8 -*-
"""
Venti – Insight IA (Notion Narrativo)
- Lector CSV robusto
- Normaliza columnas y taxonomías
- Agrupa por issues (heurística)
- KPIs placeholders
- Gráficos (Top Issues, Urgencias, Sentimientos, Canal por Issue, Urgencia vs Issue)
- Insight automático por gráfico (caption)
- Publicación opcional de assets a GitHub (raw URL)
- Página Notion con bloques narrativos, imágenes y links "Ejemplo 1/2/3"
"""

import os
import re
import json
import argparse
import shutil
import subprocess
from datetime import date
import numpy as np

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================
# Utilidades CSV / Normalización
# ================================
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
        df = pd.read_excel(csv_path, dtype=str)
        return df
    except Exception:
        pass
    raise RuntimeError(f"No pude leer el CSV (último error: {last_err})")

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
        "issue": ["issue","issue_group","issue_asignado"]
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
                  "link_a_intercom","id_intercom","fecha","rol","issue"]:
        if canon not in df.columns:
            df[canon] = ""

    return df

# ================================
# Heurística de Issues (fallback)
# ================================
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
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            return label
    return "Otros"

def build_text_base(row: pd.Series) -> str:
    parts = []
    for col in ["resumen_ia","insight_ia","palabras_clave","tema","motivo","submotivo","area"]:
        val = str(row.get(col,"") or "").strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return " | ".join(parts).lower()

# ================================
# Resumen por Issue (para narrativa)
# ================================
def top_values(series: pd.Series, n=3) -> str:
    vc = series.fillna("").replace("nan","").astype(str).str.strip().value_counts()
    items = [f"{idx} ({cnt})" for idx, cnt in vc.head(n).items() if idx]
    return ", ".join(items)

def build_issue_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for issue, grp in df.groupby("issue"):
        canales = top_values(grp["canal"])
        areas = top_values(grp["area"])
        motivos = top_values(grp["motivo"])
        submotivos = top_values(grp["submotivo"])
        ejemplos = grp.loc[grp["link_a_intercom"].astype(str).str.startswith("http"), "link_a_intercom"].head(3).astype(str).tolist()
        rows.append({
            "issue": issue,
            "casos": int(len(grp)),
            "canales_top": canales,
            "areas_top": areas,
            "motivos_top": motivos,
            "submotivos_top": submotivos,
            "ejemplos_intercom": ejemplos
        })
    out = pd.DataFrame(rows).sort_values("casos", ascending=False)
    return out

# ================================
# Gráficos + Insights
# ================================
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def chart_top_issues(df, out_dir):
    counts = df["issue"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(range(len(counts)), counts.values)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(list(counts.index))
    ax.invert_yaxis()
    ax.set_title("Top Issues")
    ax.set_xlabel("Casos")
    return _save(fig, out_dir, "top_issues")

def chart_urgencia_pie(df, out_dir):
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    vc = df["urgencia"].fillna("Sin dato").replace("nan","Sin dato").value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%")
    ax.set_title("Distribución de Urgencias")
    return _save(fig, out_dir, "urgencia_pie")

def chart_sentimiento_pie(df, out_dir):
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    vc = df["sentimiento"].fillna("Sin dato").replace("nan","Sin dato").value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%")
    ax.set_title("Distribución de Sentimientos")
    return _save(fig, out_dir, "sentimiento_pie")

def chart_canal_por_issue(df, out_dir):
    pivot = pd.crosstab(df["issue"], df["canal"])
    fig, ax = plt.subplots(figsize=(9,6))
    bottom = None
    x = range(len(pivot.index))
    for col in pivot.columns:
        vals = pivot[col].values
        if bottom is None:
            ax.bar(x, vals, label=str(col)); bottom = vals
        else:
            ax.bar(x, vals, bottom=bottom, label=str(col)); bottom = [b+v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels(list(pivot.index), rotation=45, ha="right")
    ax.set_title("Canal por Issue"); ax.set_ylabel("Casos"); ax.legend()
    return _save(fig, out_dir, "canal_por_issue")

def chart_urgencia_por_issue(df, out_dir):
    # Crosstab issue x urgencia
    pivot = pd.crosstab(df["issue"], df["urgencia"])

    if pivot.empty:
        # Grafico placeholder para no romper el flujo
        fig, ax = plt.subplots(figsize=(7,4))
        ax.text(0.5, 0.5, "Sin datos de Urgencia por Issue", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out_dir, "urgencia_por_issue")

    # Ordenar por total (top 8 issues)
    totals = pivot.sum(axis=1).sort_values(ascending=False)
    top_idx = totals.head(8).index
    pivot = pivot.loc[top_idx]

    # Orden sugerida de urgencias (si existen)
    desired_cols = ["Alta", "Media", "Baja", "Sin dato", "nan", None]
    ordered_cols = [c for c in desired_cols if c in pivot.columns] + [c for c in pivot.columns if c not in desired_cols]
    pivot = pivot[ordered_cols]

    # Stacked bars con bottom acumulado
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index), dtype=float)

    for col in pivot.columns:
        vals = pivot[col].astype(float).values
        ax.bar(x, vals, bottom=bottom, label=str(col))
        bottom = bottom + vals  # acumulado

    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot.index), rotation=45, ha="right")
    ax.set_title("Distribución de Urgencias en Top Issues")
    ax.set_ylabel("Casos")
    ax.legend()

    return _save(fig, out_dir, "urgencia_por_issue")

def insight_for_chart(kind: str, df: pd.DataFrame) -> str:
    try:
        if kind == "top_issues":
            counts = df["issue"].value_counts(normalize=True) * 100
            if not counts.empty:
                top = counts.idxmax()
                return f"El {counts[top]:.0f}% de los contactos se concentra en “{top}”."
        if kind == "urgencia_pie":
            counts = df["urgencia"].fillna("Sin dato").replace("nan","Sin dato").value_counts(normalize=True) * 100
            if not counts.empty:
                top = counts.idxmax()
                return f"La urgencia más frecuente es {top} ({counts[top]:.0f}%). Priorizar SLA para altos."
        if kind == "sentimiento_pie":
            counts = df["sentimiento"].fillna("Sin dato").replace("nan","Sin dato").value_counts(normalize=True) * 100
            if not counts.empty:
                top = counts.idxmax()
                return f"El sentimiento predominante es {top} ({counts[top]:.0f}%)."
        if kind == "canal_por_issue":
            top_issue = df["issue"].value_counts().idxmax()
            canal_counts = df.loc[df["issue"]==top_issue, "canal"].value_counts()
            if not canal_counts.empty:
                return f"Para “{top_issue}”, el canal dominante es {canal_counts.idxmax()}."
        if kind == "urgencia_por_issue":
            high = df[df["urgencia"].str.lower().eq("alta") if "urgencia" in df else []]["issue"].value_counts()
            if not high.empty:
                return f"En urgencia ALTA prevalece “{high.idxmax()}”."
    except Exception:
        pass
    return " "

# ================================
# Publicación a GitHub (assets)
# ================================
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
        files = ["top_issues.png", "urgencia_pie.png", "sentimiento_pie.png", "canal_por_issue.png", "urgencia_por_issue.png"]
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
    except subprocess.CalledProcessError as e:
        if "nothing to commit" not in (e.stderr or "") and "nothing to commit" not in (e.stdout or ""):
            raise
    _run_git(["git", "push", "origin", branch], cwd=repo_path)
    remote = _run_git(["git", "config", "--get", "remote.origin.url"], cwd=repo_path)
    owner, repo = _parse_remote_origin(remote)
    base_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/reports/{date_subdir}"
    return base_raw

# ================================
# Notion Blocks helpers
# ================================
def _rt(text: str, link: str | None = None):
    obj = {"type": "text", "text": {"content": str(text)}}
    if link:
        obj["text"]["link"] = {"url": link}
    return [obj]

def _heading(level: int, text: str):
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rt(text)}}

def _para(text: str):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}

def _bullet(text: str, link: str | None = None):
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rt(text, link)}}

def _image_external(url: str, caption: str = ""):
    blk = {"object":"block","type":"image","image":{"type":"external","external":{"url":url}}}
    if caption:
        blk["image"]["caption"] = _rt(caption)
    return blk

# ================================
# Construcción de página Notion
# ================================
def build_notion_blocks(meta: dict, df: pd.DataFrame, resumen_df: pd.DataFrame, assets: dict) -> list:
    total = len(df)
    date_str = meta.get("fecha", date.today().isoformat())
    src = meta.get("fuente","")

    # Top 3
    top3 = df["issue"].value_counts().head(3)
    top3_lines = [f"- {i} → {c} casos ({int(round(c/total*100,0))}%)" for i,c in top3.items()]

    # KPIs placeholders
    kpi_lines = [
        "Tickets resueltos: —",
        "% Primera Respuesta: —",
        "Tiempo medio de resolución: —",
        "Satisfacción (CSAT): —",
        "% Casos resueltos (Gemini): —"
    ]

    blocks = []
    blocks.append(_heading(1, f"Reporte CX – {date_str}"))

    # Resumen Ejecutivo (narrativo)
    blocks.append(_heading(2, "Resumen Ejecutivo"))
    blocks.append(_para("Durante el periodo analizado se procesaron conversaciones de Intercom para identificar patrones, problemas recurrentes y oportunidades de mejora."))
    blocks.append(_bullet(f"📅 Fecha del análisis: {date_str}"))
    blocks.append(_bullet(f"📂 Fuente de datos: {src}"))
    blocks.append(_bullet(f"💬 Conversaciones procesadas: {total}"))

    # KPIs a la vista
    blocks.append(_heading(2, "KPIs a la vista"))
    for k in kpi_lines:
        blocks.append(_bullet(k))

    # Top 3 issues
    blocks.append(_heading(2, "Top 3 issues"))
    if top3_lines:
        for ln in top3_lines:
            blocks.append(_para(ln))
    else:
        blocks.append(_para("No hay datos suficientes."))

    # Gráficos (con insights debajo)
    # Top Issues
    if assets.get("top_issues"):
        blocks.append(_heading(2, "Top Issues"))
        blocks.append(_image_external(assets["top_issues"], caption=insight_for_chart("top_issues", df)))

    # Distribución de Urgencias
    if assets.get("urgencia_pie"):
        blocks.append(_heading(2, "Distribución de Urgencias"))
        blocks.append(_image_external(assets["urgencia_pie"], caption=insight_for_chart("urgencia_pie", df)))

    # Distribución de Sentimientos
    if assets.get("sentimiento_pie"):
        blocks.append(_heading(2, "Distribución de Sentimientos"))
        blocks.append(_image_external(assets["sentimiento_pie"], caption=insight_for_chart("sentimiento_pie", df)))

    # Canal por Issue
    if assets.get("canal_por_issue"):
        blocks.append(_heading(2, "Canal por Issue"))
        blocks.append(_image_external(assets["canal_por_issue"], caption=insight_for_chart("canal_por_issue", df)))

    # Urgencia vs Issue
    if assets.get("urgencia_por_issue"):
        blocks.append(_heading(2, "Distribución de Urgencias en Top Issues"))
        blocks.append(_image_external(assets["urgencia_por_issue"], caption="Cruce de prioridad y categoría para detectar focos críticos."))

    # Issues detallados (narrativo + links)
    blocks.append(_heading(2, "Issues Detallados"))
    for _, row in resumen_df.iterrows():
        issue = row["issue"]; casos = int(row["casos"])
        blocks.append(_heading(3, f"{issue} ({casos} casos)"))
        if str(row.get("canales_top","")).strip():
            blocks.append(_bullet(f"Canales: {row['canales_top']}"))
        if str(row.get("areas_top","")).strip():
            blocks.append(_bullet(f"Áreas: {row['areas_top']}"))
        if str(row.get("motivos_top","")).strip():
            blocks.append(_bullet(f"Motivos: {row['motivos_top']}"))
        if str(row.get("submotivos_top","")).strip():
            blocks.append(_bullet(f"Submotivos: {row['submotivos_top']}"))

        ejemplos = row.get("ejemplos_intercom", [])
        if isinstance(ejemplos, str):
            ejemplos = [e.strip() for e in ejemplos.split("|") if e.strip()]
        if ejemplos:
            blocks.append(_para("Ejemplos:"))
            for i, url in enumerate(ejemplos[:3], start=1):
                if url.startswith("http"):
                    blocks.append(_bullet(f"Ejemplo {i}", link=url))

    return blocks

def notion_create_page(parent_page_id: str, token: str, page_title: str, children_blocks: list):
    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "properties": {"title": {"title": [{"type": "text", "text": {"content": page_title}}]}},
        "children": children_blocks
    }
    # FIX de sintaxis ^ la línea de arriba tenía una llave de más antes; ya está corregido:
    # Debe ser exactamente como el dict 'payload' anterior (sin llaves extra).

    # Enviar
    import requests
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

# ================================
# Pipeline principal
# ================================
def run(csv_path: str, out_dir: str,
        notion_token: str | None,
        notion_parent: str | None,
        publish_github: bool,
        github_repo_path: str | None,
        github_branch: str) -> None:

    print("▶ Script iniciado")
    print("CSV:", csv_path)
    print("OUT:", out_dir)
    print("NOTION_TOKEN:", "OK" if notion_token else "FALTA")
    print("PARENT:", notion_parent)

    # 1) Carga + normalización + issues (si falta)
    df = load_csv_robusto(csv_path)
    df = normalize_columns(df)

    # Completar/asegurar columna 'issue'
    if "issue" not in df.columns or (df["issue"].fillna("").str.strip() == "").all():
        df["texto_base"] = df.apply(build_text_base, axis=1)
        df["issue"] = df["texto_base"].apply(assign_issue_group)

    # 2) Resumen por issue (para narrativa)
    resumen_df = build_issue_summary(df)

    # 3) Gráficos a disco
    os.makedirs(out_dir, exist_ok=True)
    p_top = chart_top_issues(df, out_dir)
    p_urg_pie = chart_urgencia_pie(df, out_dir)
    p_sent_pie = chart_sentimiento_pie(df, out_dir)
    p_canal_issue = chart_canal_por_issue(df, out_dir)
    p_urg_issue = chart_urgencia_por_issue(df, out_dir)

    # 4) (Opcional) publicar a GitHub para URL pública
    assets_base = None
    if publish_github and github_repo_path:
        try:
            assets_base = publish_images_to_github(
                out_dir=out_dir,
                repo_path=github_repo_path,
                branch=github_branch,
                date_subdir=date.today().isoformat(),
                files=[
                    os.path.basename(p_top),
                    os.path.basename(p_urg_pie),
                    os.path.basename(p_sent_pie),
                    os.path.basename(p_canal_issue),
                    os.path.basename(p_urg_issue),
                ],
            )
            print(f"🌐 Assets publicados en: {assets_base}")
        except Exception as e:
            print(f"⚠️ No pude publicar en GitHub: {e}")

    # 5) Construir URLs públicas o usar paths locales (Notion requiere URL)
    def to_url(local_path):
        if assets_base:
            return f"{assets_base}/{os.path.basename(local_path)}"
        # Si no hay URL pública, igual enviamos file:// (Notion NO la usará)
        return f"file://{os.path.abspath(local_path)}"

    assets = {
        "top_issues": to_url(p_top),
        "urgencia_pie": to_url(p_urg_pie),
        "sentimiento_pie": to_url(p_sent_pie),
        "canal_por_issue": to_url(p_canal_issue),
        "urgencia_por_issue": to_url(p_urg_issue),
    }

    # 6) Notion
    if notion_token and notion_parent:
        meta = {"fecha": date.today().isoformat(), "fuente": os.path.basename(csv_path)}
        blocks = build_notion_blocks(meta, df, resumen_df, assets)
        notion_create_page(notion_parent, notion_token, f"Reporte CX – {date.today().isoformat()}", blocks)
        print("✅ Publicado en Notion.")
    else:
        print("ℹ️ Sin credenciales de Notion: se generaron solo archivos locales.")
        print(json.dumps({
            "top_issues": p_top,
            "urgencia_pie": p_urg_pie,
            "sentimiento_pie": p_sent_pie,
            "canal_por_issue": p_canal_issue,
            "urgencia_por_issue": p_urg_issue
        }, indent=2, ensure_ascii=False))

# ================================
# CLI
# ================================
def main():
    ap = argparse.ArgumentParser(description="Venti – Insight IA Notion narrativo")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de conversaciones")
    ap.add_argument("--out", default="./salida", help="Directorio de salida (default ./salida)")
    ap.add_argument("--notion_token", default=os.getenv("NOTION_TOKEN"), help="Token de Notion")
    ap.add_argument("--notion_parent", default=os.getenv("NOTION_PARENT_PAGE_ID"), help="ID de página padre (Notion)")
    ap.add_argument("--publish_github", action="store_true", help="Publicar PNGs a GitHub")
    ap.add_argument("--github_repo_path", default=None, help="Ruta local al repo clonado")
    ap.add_argument("--github_branch", default="main", help="Branch a usar (default main)")
    args = ap.parse_args()

    run(
        csv_path=args.csv,
        out_dir=args.out,
        notion_token=args.notion_token,
        notion_parent=args.notion_parent,
        publish_github=args.publish_github,
        github_repo_path=args.github_repo_path,
        github_branch=args.github_branch
    )

if __name__ == "__main__":
    main()


