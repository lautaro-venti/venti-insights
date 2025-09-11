# -*- coding: utf-8 -*-
"""
Venti · Validación por Hora → Notion (estilo Metabase)
Requisitos: pip install pandas requests python-dateutil
"""

import sys, argparse, json, time, unicodedata
from datetime import datetime
import pandas as pd
import requests

NOTION_VERSION = "2022-06-28"
API_BASE = "https://api.notion.com/v1"
_ADDED_ALIASES = {}

# ---------- Helpers ----------
def _headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return " ".join(s.split())

def load_csv_robusto(path: str) -> pd.DataFrame:
    encs = ["utf-8-sig","utf-8","cp1252","latin-1","utf-16","utf-16-le","utf-16-be"]
    seps = [",",";","\t", None]
    last = None
    for enc in encs:
        for sep in seps:
            try:
                if sep is None:
                    return pd.read_csv(path, encoding=enc, sep=None, engine="python", dtype=str, on_bad_lines="skip")
                else:
                    return pd.read_csv(path, encoding=enc, sep=sep, engine="python", dtype=str, on_bad_lines="skip")
            except Exception as e:
                last = e
                continue
    raise RuntimeError(f"No pude leer el CSV ({path}). Último error: {last}")

def parse_dt_tolerante(s: str):
    if s is None: return None
    txt = str(s).strip().lower()
    if not txt: return None
    meses = {"enero":"01","febrero":"02","marzo":"03","abril":"04","mayo":"05","junio":"06","julio":"07","agosto":"08","septiembre":"09","setiembre":"09","octubre":"10","noviembre":"11","diciembre":"12"}
    dias = ["lunes","martes","miércoles","miercoles","jueves","viernes","sábado","sabado","domingo"]
    for d in dias:
        if txt.startswith(d + ","): txt = txt[len(d)+1:].strip()
    txt = txt.replace(" de ", " ").replace(",", " ")
    for m,n in meses.items():
        txt = txt.replace(" "+m+" ", f" {n} ")
    txt = " ".join(txt.split())
    cands = [txt, txt.replace(" ","-"), txt.replace(" ","/")]
    fmts = ["%d %m %Y %H:%M","%d-%m-%Y-%H:%M","%d/%m/%Y/%H:%M","%Y-%m-%d %H:%M","%Y/%m/%d %H:%M","%d/%m/%Y %H:%M","%d-%m-%Y %H:%M","%d/%m/%Y %H","%Y-%m-%d %H","%d-%m-%Y %H"]
    for c in cands:
        try:
            dt = pd.to_datetime(c, dayfirst=True, errors="raise")
            return dt.tz_localize(None) if hasattr(dt,"tz_localize") else dt
        except Exception:
            pass
    for c in cands:
        for f in fmts:
            try:
                return datetime.strptime(c, f)
            except Exception:
                continue
    import re
    hhmm = re.search(r"(\d{1,2}:\d{2})", txt)
    dmy  = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", txt)
    if dmy:
        d,m,y = dmy.groups(); y = ("20"+y) if len(y)==2 else y
        h = hhmm.group(1) if hhmm else "00:00"
        try:
            return datetime.strptime(f"{d.zfill(2)}/{m.zfill(2)}/{y} {h}", "%d/%m/%Y %H:%M")
        except Exception:
            pass
    return None

# ---------- Notion ----------
def get_database(token: str, db_id: str) -> dict:
    r = requests.get(f"{API_BASE}/databases/{db_id}", headers=_headers(token))
    if not r.ok: raise RuntimeError(f"Error leyendo DB: {r.status_code} {r.text}")
    return r.json()

def get_title_prop_name(db_json: dict):
    for name, spec in (db_json.get("properties") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "title":
            return name
    return None

def ensure_db(token: str, parent_page_id: str, db_id: str|None) -> str:
    if db_id: return db_id
    payload = {
        "parent": {"type":"page_id","page_id": parent_page_id},
        "title": [{"type":"text","text":{"content":"Validación por Hora (Todos los Eventos)"}}],
        "properties": {
            "Event Name": {"title": {}},
            "Es Hora Pico": {"checkbox": {}},
            "Event Date": {"date": {}},
            "Event ID": {"number": {"format":"number"}},
            "Hora": {"number": {"format":"number"}},
            "Primera Hora del Validador": {"rich_text": {}},
            "Row Key": {"rich_text": {}},
            "Status": {"select": {}},
            "Status (text)": {"rich_text": {}},
            "Total Evento": {"number": {"format":"number"}},
            "Total Hora (Evento)": {"number": {"format":"number"}},
            "Total Validado": {"number": {"format":"number"}},
            "Usuarios de Validadores": {"email": {}},
            "Validador – Entradas en la hora": {"number": {"format":"number"}},
            "Validation Time": {"date": {}},
        }
    }
    r = requests.post(f"{API_BASE}/databases", headers=_headers(token), data=json.dumps(payload))
    if not r.ok: raise RuntimeError(f"Error creando DB: {r.status_code} {r.text}")
    return r.json()["id"]

def ensure_schema(token: str, db_id: str, need_rowkey: bool=True):
    db = get_database(token, db_id)
    props = db.get("properties") or {}
    existing_by_norm = {_norm(n): n for n in props.keys()}
    title_prop = get_title_prop_name(db)
    if title_prop: _ADDED_ALIASES["Event Name"] = title_prop
    wanted = {
        "Es Hora Pico": {"checkbox": {}},
        "Event Date": {"date": {}},
        "Event ID": {"number": {"format":"number"}},
        "Hora": {"number": {"format":"number"}},
        "Primera Hora del Validador": {"rich_text": {}},
        "Row Key": {"rich_text": {}},
        "Status": {"select": {}},
        "Status (text)": {"rich_text": {}},
        "Total Evento": {"number": {"format":"number"}},
        "Total Hora (Evento)": {"number": {"format":"number"}},
        "Total Validado": {"number": {"format":"number"}},
        "Usuarios de Validadores": {"email": {}},
        "Validador – Entradas en la hora": {"number": {"format":"number"}},
        "Validation Time": {"date": {}},
    }
    if not need_rowkey: wanted.pop("Row Key", None)

    for want_name, spec in wanted.items():
        want_norm = _norm(want_name)
        real = existing_by_norm.get(want_norm)
        if real:
            if props[real].get("type") == list(spec.keys())[0]:
                _ADDED_ALIASES[want_name] = real
                continue
            alt = f"{want_name} (text)"
            r = requests.patch(f"{API_BASE}/databases/{db_id}", headers=_headers(token),
                               data=json.dumps({"properties": {alt: spec}}))
            if r.ok:
                _ADDED_ALIASES[want_name] = alt
                props[alt] = spec
                existing_by_norm[_norm(alt)] = alt
                continue
            raise RuntimeError(f"Error creando alias para '{want_name}': {r.status_code} {r.text}")
        r = requests.patch(f"{API_BASE}/databases/{db_id}", headers=_headers(token),
                           data=json.dumps({"properties": {want_name: spec}}))
        if r.ok:
            _ADDED_ALIASES[want_name] = want_name
            props[want_name] = spec
            existing_by_norm[want_norm] = want_name
            continue
        if r.status_code == 400 and "Cannot update property" in (r.text or ""):
            alt = f"{want_name} (text)"
            r2 = requests.patch(f"{API_BASE}/databases/{db_id}", headers=_headers(token),
                                data=json.dumps({"properties": {alt: spec}}))
            if r2.ok:
                _ADDED_ALIASES[want_name] = alt
                props[alt] = spec
                existing_by_norm[_norm(alt)] = alt
                continue
        raise RuntimeError(f"Error creando propiedad '{want_name}': {r.status_code} {r.text}")

def alias(name: str) -> str:
    return _ADDED_ALIASES.get(name, name)

# ---------- Transformaciones ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        cc = _norm(c)
        if cc in ("event name","evento","nombre evento"): mapping[c] = "Event Name"
        elif cc in ("temp id","event id","id evento","temp_id","tempid"): mapping[c] = "Event ID"
        elif cc in ("temp validation time","validation time","fecha hora","timestamp"): mapping[c] = "Validation Time"
        elif cc in ("status","estado"): mapping[c] = "Status"
        elif cc in ("total validado","total","validados","tickets","entradas"): mapping[c] = "Total Validado"
        elif cc in ("temp usuarios de validadores","usuarios de validadores","validador","email","validator email","usuario"): mapping[c] = "Usuarios de Validadores"
    return df.rename(columns=mapping) if mapping else df

def add_computed_fields(df: pd.DataFrame, peak_pct: float = 0.9, tz_name: str | None = None) -> pd.DataFrame:
    # Coerce básicos
    if "Event ID" in df.columns:
        df["Event ID"] = pd.to_numeric(df["Event ID"], errors="coerce")

    if "Total Validado" in df.columns:
        df["Total Validado"] = pd.to_numeric(df["Total Validado"], errors="coerce").fillna(0).astype(int)
    else:
        df["Total Validado"] = 0

    df["Usuarios de Validadores"] = (
        df.get("Usuarios de Validadores", "")
          .astype(str).str.strip().str.lower()
          .replace({"nan": ""})
    )

    # Parse fecha/hora
    if "Validation Time" not in df.columns:
        raise RuntimeError("Falta la columna 'Validation Time' en el CSV.")
    parsed = pd.to_datetime(df["Validation Time"].apply(parse_dt_tolerante), errors="coerce")

    df["Validation Time"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S").where(parsed.notna(), None)
    df["Event Date"] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
    df["Hora"] = parsed.dt.hour.where(parsed.notna(), None)

    # --- Claves normalizadas para agregados (evitan el error de tipos) ---
    df["_EID_K"]   = df["Event ID"].fillna(-1).astype("Int64")
    df["_EDATE_K"] = df["Event Date"].fillna("NA").astype(str)
    df["_HORA_K"]  = df["Hora"].fillna(-1).astype("Int64")
    df["_MAIL_K"]  = df["Usuarios de Validadores"].fillna("").astype(str)

    # Validador – Entradas en la hora
    key_uvh = ["_EID_K","_EDATE_K","_HORA_K","_MAIL_K"]
    val_por_val_hora = (
        df.groupby(key_uvh, dropna=False)["Total Validado"]
          .sum().rename("Validador – Entradas en la hora").reset_index()
    )

    # Total Hora (Evento)
    key_evh = ["_EID_K","_EDATE_K","_HORA_K"]
    total_hora_evento = (
        val_por_val_hora.groupby(key_evh, dropna=False)["Validador – Entradas en la hora"]
                        .sum().rename("Total Hora (Evento)").reset_index()
    )

    # Total Evento
    key_ev = ["_EID_K","_EDATE_K"]
    total_evento = (
        df.groupby(key_ev, dropna=False)["Total Validado"]
          .sum().rename("Total Evento").reset_index()
    )

    # Merge sobre claves normalizadas
    for c in ["Validador – Entradas en la hora","Total Hora (Evento)","Total Evento"]:
        if c in df.columns: df = df.drop(columns=[c])
    df = df.merge(val_por_val_hora, on=key_uvh, how="left")
    df = df.merge(total_hora_evento, on=key_evh, how="left")
    df = df.merge(total_evento, on=key_ev, how="left")

    # Hora pico dentro de cada evento/día
    max_hora = df.groupby(key_ev, dropna=False)["Total Hora (Evento)"].transform("max")
    df["Es Hora Pico"] = (
        (df["Total Hora (Evento)"] >= (peak_pct * max_hora)) &
        df["Total Hora (Evento)"].notna() &
        (df["Total Hora (Evento)"] > 0)
    )

    # Row Key estable: eventid_fecha_hora_email
    def _rk(row):
        ev = "NA" if pd.isna(row["Event ID"]) else str(int(row["Event ID"]))
        dt = row["Event Date"] or "NA"
        hr = "NA" if pd.isna(row["Hora"]) else str(int(row["Hora"]))
        em = (row.get("Usuarios de Validadores") or "NA").strip().lower() or "NA"
        return f"{ev}_{dt}_{hr}_{em}"
    df["Row Key"] = df.apply(_rk, axis=1)

    # Primera hora del validador (texto para vista)
    first_hour = df.groupby(["_EID_K","_EDATE_K","_MAIL_K"])["_HORA_K"].transform("min")
    df["Primera Hora del Validador"] = first_hour.astype("Int64").astype(str).where(first_hour.notna(), "")

    # limpiar columnas clave auxiliares
    return df.drop(columns=["_EID_K","_EDATE_K","_HORA_K","_MAIL_K"])

# ---------- Notion upsert ----------
def query_by_rowkey(token: str, db_id: str, row_key: str):
    payload = {"filter":{"property": alias("Row Key"), "rich_text":{"equals": row_key}}, "page_size":1}
    r = requests.post(f"{API_BASE}/databases/{db_id}/query", headers=_headers(token), data=json.dumps(payload))
    if r.status_code == 400 and "Could not find property" in (r.text or ""): return "__NO_ROWKEY__"
    if not r.ok: raise RuntimeError(f"Error query DB: {r.status_code} {r.text}")
    res = r.json().get("results", [])
    return res[0]["id"] if res else None

def props_from_row(row: dict) -> dict:
    def _title(s):  s = "" if s is None or (isinstance(s,float) and pd.isna(s)) else str(s); return {"title":[{"type":"text","text":{"content": s or "—"}}]}
    def _rt(s):     return {"rich_text":[{"type":"text","text":{"content": str(s)}}]} if (s not in [None,""] and not (isinstance(s,float) and pd.isna(s))) else {"rich_text":[]}
    def _sel(s):    return {"select":{"name": str(s)}} if (s not in [None,""] and not (isinstance(s,float) and pd.isna(s))) else {"select": None}
    def _num(x):    return {"number": None if (x is None or (isinstance(x,float) and pd.isna(x))) else float(x)}
    def _date(iso): return {"date": None} if (iso in [None,""] or (isinstance(iso,float) and pd.isna(iso))) else {"date":{"start": str(iso)}}
    def _email(x):  return {"email": (None if (x in [None,""] or (isinstance(x,float) and pd.isna(x))) else str(x).strip().lower())}
    def _chk(b):    return {"checkbox": bool(False if (b is None or (isinstance(b,float) and pd.isna(b))) else b)}

    return {
        alias("Event Name"): _title(row.get("Event Name")),
        alias("Es Hora Pico"): _chk(row.get("Es Hora Pico")),
        alias("Event Date"): _date(row.get("Event Date")),
        alias("Event ID"): _num(row.get("Event ID")),
        alias("Hora"): _num(row.get("Hora")),
        alias("Primera Hora del Validador"): _rt(row.get("Primera Hora del Validador")),
        alias("Row Key"): _rt(row.get("Row Key")),
        alias("Status"): _sel(row.get("Status")),
        alias("Status (text)"): _rt(row.get("Status")),
        alias("Total Evento"): _num(row.get("Total Evento")),
        alias("Total Hora (Evento)"): _num(row.get("Total Hora (Evento)")),
        alias("Total Validado"): _num(row.get("Total Validado")),
        alias("Usuarios de Validadores"): _email(row.get("Usuarios de Validadores")),
        alias("Validador – Entradas en la hora"): _num(row.get("Validador – Entradas en la hora")),
        alias("Validation Time"): _date(row.get("Validation Time")),
    }

def upsert_page(token: str, db_id: str, props: dict, sleep_s: float = 0.15):
    rk = (props.get(alias("Row Key")) or {}).get("rich_text", [])
    rk = rk[0]["text"]["content"] if rk else ""
    pid = query_by_rowkey(token, db_id, rk) if rk else None
    if pid == "__NO_ROWKEY__": pid = None
    if pid:
        r = requests.patch(f"{API_BASE}/pages/{pid}", headers=_headers(token), data=json.dumps({"properties": props}))
    else:
        r = requests.post(f"{API_BASE}/pages", headers=_headers(token),
                          data=json.dumps({"parent":{"database_id": db_id}, "properties": props}))
    if not r.ok: raise RuntimeError(f"Error upsert page: {r.status_code} {r.text}")
    time.sleep(sleep_s)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Validación por Hora → Notion (estilo Metabase)")
    ap.add_argument("--csv", required=True, help="Ruta al CSV exportado")
    ap.add_argument("--notion_token", required=True, help="Notion integration secret")
    ap.add_argument("--db", default="", help="Database ID (si ya existe)")
    ap.add_argument("--notion_parent", default="", help="Page ID (para crear DB dentro de esa página)")
    ap.add_argument("--chunk", type=int, default=800, help="Filas por lote")
    ap.add_argument("--peak_pct", type=float, default=0.90, help="Percentil Hora Pico (0–1)")
    ap.add_argument("--tz", default="", help="Timezone (opcional)")
    args = ap.parse_args()

    if not (args.db or args.notion_parent):
        print("ERROR: pasá --db o --notion_parent."); sys.exit(2)

    df = load_csv_robusto(args.csv)
    df = normalize_columns(df)
    df = add_computed_fields(df, peak_pct=args.peak_pct, tz_name=(args.tz or None))

    database_id = ensure_db(args.notion_token, args.notion_parent or None, args.db or None)
    print(f"Usando Database ID: {database_id}")
    ensure_schema(args.notion_token, database_id, need_rowkey=True)

    recs = df.to_dict(orient="records")
    total = len(recs)
    for i in range(0, total, args.chunk):
        for row in recs[i:i+args.chunk]:
            upsert_page(args.notion_token, database_id, props_from_row(row))
        print(f"Subido: {min(i+args.chunk, total)}/{total}")
    print("✅ Listo. Revisá la base en Notion.")

if __name__ == "__main__":
    main()
