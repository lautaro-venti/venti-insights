# src/enrichment.py
import re, unicodedata
import pandas as pd
import numpy as np

def norm_txt(s: str) -> str:
    if pd.isna(s): 
        return ""
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s\-_/.:#@]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_event_catalog_from_calendar(cal_df: pd.DataFrame,
                                      col_event_name: str,
                                      col_producer: str|None=None) -> dict:
    cat = {}
    df = cal_df[[col_event_name] + ([col_producer] if col_producer and col_producer in cal_df.columns else [])]\
            .dropna().drop_duplicates()
    df["_event_name_norm"] = df[col_event_name].map(norm_txt)
    for _, row in df.iterrows():
        raw = row[col_event_name]
        nrm = row["_event_name_norm"]
        prod = row[col_producer] if col_producer and col_producer in row else ""
        if nrm:
            cat[nrm] = {"raw": raw, "producer": prod}
    return cat

def infer_event_from_resumen(resumen_series: pd.Series, event_catalog: dict, threshold_tokens=2) -> pd.DataFrame:
    names, scores = [], []
    keys = list(event_catalog.keys())
    for txt in resumen_series.fillna("").astype(str).tolist():
        t = norm_txt(txt)
        best_key, best_score = None, 0
        for k in keys:
            tokens = [w for w in k.split() if len(w) >= 4]
            hit = sum(1 for w in tokens if w in t)
            if hit > best_score:
                best_score, best_key = hit, k
        if best_score >= threshold_tokens:
            names.append(event_catalog[best_key]["raw"])
            scores.append(best_score)
        else:
            names.append("")
            scores.append(0)
    return pd.DataFrame({"event_name_inferred": names, "event_match_score": scores})

def weekly_event_kpis(conversations: pd.DataFrame,
                      calendar: pd.DataFrame,
                      col_conv_date: str,
                      col_conv_resumen: str,
                      cal_event_name_col: str,
                      cal_producer_col: str|None,
                      cal_tickets_col: str,
                      cal_gross_col: str,
                      tz="America/Argentina/Buenos_Aires") -> pd.DataFrame:
    # fechas → semana (Lunes)
    conv = conversations.copy()
    conv["_fecha"] = pd.to_datetime(conv[col_conv_date], errors="coerce", utc=True)
    if conv["_fecha"].isna().mean() > 0.6:  # fallback por si viene dayfirst
        conv["_fecha"] = pd.to_datetime(conv[col_conv_date], errors="coerce", dayfirst=True)
    conv["week_start"] = conv["_fecha"].dt.to_period("W-MON").dt.start_time

    cal = calendar.copy()
    cal["_event_datetime"] = pd.to_datetime(cal["Fecha Evento"], errors="coerce", dayfirst=True) \
                             if "Fecha Evento" in cal.columns else pd.NaT
    cal["week_start"] = cal["_event_datetime"].dt.to_period("W-MON").dt.start_time

    # catálogo y matching
    catalog = build_event_catalog_from_calendar(cal, cal_event_name_col, cal_producer_col)
    inf = infer_event_from_resumen(conv[col_conv_resumen], catalog, threshold_tokens=2)
    conv = pd.concat([conv, inf], axis=1)

    # soporte semanal por evento (confianza)
    CONF = 2
    conv["_event_for_rate"] = np.where(conv["event_match_score"]>=CONF, conv["event_name_inferred"], "")
    conv_week = conv[conv["week_start"].notna()]
    grp_conv = (conv_week.groupby(["week_start","_event_for_rate"])
                .size().rename("conv_count").reset_index()
                .rename(columns={"_event_for_rate":"event_name"}))

    # ventas semanales por evento
    def _to_num(x): 
        return pd.to_numeric(str(x).replace(".","").replace(",", "."), errors="coerce")
    cal["_tickets_sold"] = cal[cal_tickets_col].map(_to_num) if cal_tickets_col in cal.columns else 0
    cal["_gross_ars"]    = cal[cal_gross_col].map(_to_num)    if cal_gross_col  in cal.columns else 0
    cal["_producer"]     = cal[cal_producer_col] if cal_producer_col and cal_producer_col in cal.columns else ""
    sales_grp = (cal[cal["week_start"].notna()]
                 .groupby(["week_start", cal_event_name_col], as_index=False)
                 .agg(tickets_sold=("_tickets_sold","sum"),
                      gross_sales_ars=("_gross_ars","sum"),
                      producer=("_producer","first"))
                 .rename(columns={cal_event_name_col:"event_name"}))
    # join ventas + soporte
    kpis = sales_grp.merge(grp_conv, on=["week_start","event_name"], how="left")
    kpis["conv_count"] = kpis["conv_count"].fillna(0).astype(int)
    kpis["problem_rate"] = np.where(kpis["tickets_sold"]>0, kpis["conv_count"]/kpis["tickets_sold"], np.nan)
    return kpis, conv
