# src/enrichment.py
import re
import unicodedata
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


def norm_txt(s: str) -> str:
    """
    Normaliza texto para matching:
    - lower
    - strip
    - quita tildes
    - deja solo [a-z0-9\s\-_/.:#@]
    - colapsa espacios
    """
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = "".join(
        c
        for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    s = re.sub(r"[^a-z0-9\s\-_/.:#@]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_event_catalog_from_calendar(
    cal_df: pd.DataFrame,
    col_event_name: str,
    col_producer: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Construye un catálogo {event_name_normalizado: {raw, producer}} desde el calendario.
    Sirve para luego matchear contra el resumen de la conversación.
    """
    cat: Dict[str, Dict[str, Any]] = {}

    cols = [col_event_name]
    if col_producer and col_producer in cal_df.columns:
        cols.append(col_producer)

    df = cal_df[cols].dropna().drop_duplicates()
    df["_event_name_norm"] = df[col_event_name].map(norm_txt)

    for _, row in df.iterrows():
        raw = row[col_event_name]
        nrm = row["_event_name_norm"]
        prod = ""
        if col_producer and col_producer in row:
            prod = row[col_producer]
        if nrm:
            cat[nrm] = {"raw": raw, "producer": prod}

    return cat


def infer_event_from_resumen(
    resumen_series: pd.Series,
    event_catalog: Dict[str, Dict[str, Any]],
    threshold_tokens: int = 2,
) -> pd.DataFrame:
    """
    Para cada resumen de conversación intenta inferir el evento usando el catálogo.
    - Cuenta cuántos tokens (>=4 letras) del nombre de evento matchean en el texto normalizado.
    - Si 'hits' >= threshold_tokens, asigna ese evento.
    Devuelve DataFrame con columnas: event_name_inferred, event_match_score.
    """
    names = []
    scores = []
    keys = list(event_catalog.keys())

    for txt in resumen_series.fillna("").astype(str).tolist():
        t = norm_txt(txt)
        best_key = None
        best_score = 0

        for k in keys:
            tokens = [w for w in k.split() if len(w) >= 4]
            hit = sum(1 for w in tokens if w in t)
            if hit > best_score:
                best_score = hit
                best_key = k

        if best_score >= threshold_tokens and best_key is not None:
            names.append(event_catalog[best_key]["raw"])
            scores.append(best_score)
        else:
            names.append("")
            scores.append(0)

    return pd.DataFrame(
        {"event_name_inferred": names, "event_match_score": scores}
    )


def weekly_event_kpis(
    conversations: pd.DataFrame,
    calendar: pd.DataFrame,
    col_conv_date: str,
    col_conv_resumen: str,
    cal_event_name_col: str,
    cal_producer_col: str | None,
    cal_tickets_col: str,
    cal_gross_col: str,
    tz: str = "America/Argentina/Buenos_Aires",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula KPIs semanales por evento:
    - tickets_sold, gross_sales_ars (desde calendario)
    - conv_count (desde Intercom, solo si el match de evento tiene score >= CONF)
    - problem_rate = conv_count / tickets_sold

    Devuelve:
      kpis_df, conversations_enriched_df
    """
    conv = conversations.copy()

    # Fecha de conversación → semana (lunes)
    conv["_fecha"] = pd.to_datetime(
        conv[col_conv_date], errors="coerce", utc=True
    )
    # Fallback por si vienen dayfirst
    if conv["_fecha"].isna().mean() > 0.6:
        conv["_fecha"] = pd.to_datetime(
            conv[col_conv_date], errors="coerce", dayfirst=True
        )
    conv["week_start"] = conv["_fecha"].dt.to_period("W-MON").dt.start_time

    # Fecha de evento → semana (lunes)
    cal = calendar.copy()
    if "Fecha Evento" in cal.columns:
        cal["_event_datetime"] = pd.to_datetime(
            cal["Fecha Evento"], errors="coerce", dayfirst=True
        )
        cal["week_start"] = cal["_event_datetime"].dt.to_period(
            "W-MON"
        ).dt.start_time
    else:
        cal["_event_datetime"] = pd.NaT
        cal["week_start"] = pd.NaT

    # Catálogo de eventos y matching
    catalog = build_event_catalog_from_calendar(
        cal, cal_event_name_col, cal_producer_col
    )
    inf = infer_event_from_resumen(
        conv[col_conv_resumen], catalog, threshold_tokens=2
    )
    conv = pd.concat([conv, inf], axis=1)

    # Soporte semanal por evento (sólo si confianza >= CONF)
    CONF = 2
    conv["_event_for_rate"] = np.where(
        conv["event_match_score"] >= CONF,
        conv["event_name_inferred"],
        "",
    )
    conv_week = conv[conv["week_start"].notna()]
    grp_conv = (
        conv_week.groupby(["week_start", "_event_for_rate"])
        .size()
        .rename("conv_count")
        .reset_index()
        .rename(columns={"_event_for_rate": "event_name"})
    )

    # Ventas semanales por evento
    def _to_num(x):
        return pd.to_numeric(
            str(x).replace(".", "").replace(",", "."),
            errors="coerce",
        )

    if cal_tickets_col in cal.columns:
        cal["_tickets_sold"] = cal[cal_tickets_col].map(_to_num)
    else:
        cal["_tickets_sold"] = 0

    if cal_gross_col in cal.columns:
        cal["_gross_ars"] = cal[cal_gross_col].map(_to_num)
    else:
        cal["_gross_ars"] = 0

    if cal_producer_col and cal_producer_col in cal.columns:
        cal["_producer"] = cal[cal_producer_col]
    else:
        cal["_producer"] = ""

    sales_grp = (
        cal[cal["week_start"].notna()]
        .groupby(["week_start", cal_event_name_col], as_index=False)
        .agg(
            tickets_sold=("_tickets_sold", "sum"),
            gross_sales_ars=("_gross_ars", "sum"),
            producer=("_producer", "first"),
        )
        .rename(columns={cal_event_name_col: "event_name"})
    )

    # Join ventas + soporte
    kpis = sales_grp.merge(
        grp_conv, on=["week_start", "event_name"], how="left"
    )
    kpis["conv_count"] = kpis["conv_count"].fillna(0).astype(int)
    kpis["problem_rate"] = np.where(
        kpis["tickets_sold"] > 0,
        kpis["conv_count"] / kpis["tickets_sold"],
        np.nan,
    )

    return kpis, conv