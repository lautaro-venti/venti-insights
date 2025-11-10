# -*- coding: utf-8 -*-
"""
Venti · Validación por Hora → Notion (semanal, SOLO relación a validadores)
- Crea SIEMPRE una database nueva dentro de --notion_parent (una por semana/CSV)
- Fechas ES, shift_hours
- Relación "Costo Por Evento" (por Event ID)
- Relación "Usuarios de Validadores" (por email) con create-if-missing
- Opción --only_with_costs: sube solo filas con costo relacionado
- NO guarda la propiedad email "Usuarios de Validadores" en la DB semanal
Requisitos: pip install pandas requests
"""

import argparse, json, time, unicodedata, re, os
from datetime import datetime, timedelta
from collections import OrderedDict

import pandas as pd
import requests

NOTION_VERSION = "2022-06-28"
API_BASE = "https://api.notion.com/v1"
_ADDED_ALIASES = {}
_REL_CACHE = {}          # cache EventID -> page_id (Costos)
_VALIDATOR_CACHE = {}    # cache email -> page_id (Validadores)

# ----------------- HTTP / helpers -----------------
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

# números robustos
def num_from_any(x, int_out=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.count(",") > 0 and s.count(".") > 0:
        last = max(s.rfind(","), s.rfind("."))
        int_part = re.sub(r"[.,]", "", s[:last])
        frac_part = s[last+1:]
        s = int_part + "." + frac_part
    elif s.count(",") > 0:
        parts = s.split(",")
        if len(parts) > 1 and len(parts[-1]) == 2:
            s = "".join(parts[:-1]) + "." + parts[-1]
        else:
            s = "".join(parts)
    try:
        val = float(s)
        return int(round(val)) if int_out else val
    except Exception:
        return None

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

# ----------------- Parse de fechas ES -----------------
_DIAS = ["lunes","martes","miércoles","miercoles","jueves","viernes","sábado","sabado","domingo"]
_MESES = {
    "enero":"01","febrero":"02","marzo":"03","abril":"04","mayo":"05","junio":"06",
    "julio":"07","agosto":"08","septiembre":"09","setiembre":"09","octubre":"10",
    "noviembre":"11","diciembre":"12"
}

def parse_dt_es(s: str) -> datetime | None:
    if s is None: return None
    raw = str(s).strip().lower()
    if not raw: return None
    for d in _DIAS:
        if raw.startswith(d + ","):
            raw = raw[len(d)+1:].strip()
            break
    txt = raw.replace(" de ", " ").replace(",", " ")
    txt = " ".join(txt.split())

    m = re.match(
        r"^(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(\d{1,2})\s+(\d{4})(?:\s+(\d{1,2}):(\d{2}))?$",
        txt
    )
    if m:
        mes_txt, d, y, hh, mm = m.groups()
        mnum = _MESES[mes_txt]
        hh = hh or "00"; mm = mm or "00"
        try:
            return datetime.strptime(f"{d.zfill(2)}/{mnum}/{y} {hh}:{mm}", "%d/%m/%Y %H:%M")
        except Exception:
            return None

    tmp = txt
    for mes_txt, mnum in _MESES.items():
        tmp = re.sub(rf"\b{mes_txt}\b", mnum, tmp)

    candidates = [tmp, tmp.replace(" ", "-"), tmp.replace(" ", "/")]
    fmts = [
        "%d %m %Y %H:%M", "%d-%m-%Y-%H:%M", "%d/%m/%Y/%H:%M",
        "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H", "%Y-%m-%d %H", "%d-%m-%Y %H",
    ]
    for c in candidates:
        try:
            dt = pd.to_datetime(c, dayfirst=True, errors="raise")
            return dt.to_pydatetime().replace(tzinfo=None)
        except Exception:
            pass
    for c in candidates:
        for f in fmts:
            try:
                return datetime.strptime(c, f)
            except Exception:
                continue

    m_dmy = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", tmp)
    if m_dmy:
        d, mnum, y = m_dmy.groups()
        y = "20"+y if len(y) == 2 else y
        m_hm = re.search(r"(\d{1,2}):(\d{2})", tmp)
        hh, mm = (m_hm.groups() if m_hm else ("00","00"))
        try:
            return datetime.strptime(f"{d.zfill(2)}/{mnum.zfill(2)}/{y} {hh}:{mm}", "%d/%m/%Y %H:%M")
        except Exception:
            return None
    return None

# ----------------- Notion helpers -----------------
def get_database(token: str, db_id: str) -> dict:
    r = requests.get(f"{API_BASE}/databases/{db_id}", headers=_headers(token))
    if not r.ok:
        raise RuntimeError(f"Error leyendo DB: {r.status_code} {r.text}")
    return r.json()

def get_title_prop_name(db_json: dict):
    for name, spec in (db_json.get("properties") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "title":
            return name
    return None

def find_first_email_prop_name(db_json: dict) -> str | None:
    props = db_json.get("properties") or {}
    email_props = [n for n, p in props.items() if p.get("type") == "email"]
    if not email_props:
        return None
    email_props_sorted = sorted(email_props, key=lambda n: (0 if "contact" in _norm(n) else (1 if "email" in _norm(n) else 2), n))
    return email_props_sorted[0]

def ensure_db(token: str, parent_page_id: str,
              rel_costs_db: str | None = None, rel_costs_prop_name: str = "Costo Por Evento",
              rel_validators_db: str | None = None, rel_validators_prop_name: str = "Usuarios de Validadores") -> str:
    """Crea la DB semanal SIN propiedad email; sólo relaciones."""
    props = OrderedDict()
    props["Event Name"] = {"title": {}}
    if rel_costs_db:
        props[rel_costs_prop_name] = {"type": "relation", "relation": {"database_id": rel_costs_db, "single_property": {}}}
    # NO creamos email aquí
    if rel_validators_db:
        props[rel_validators_prop_name] = {"type": "relation", "relation": {"database_id": rel_validators_db, "single_property": {}}}
    props["Validation Time"] = {"date": {}}
    props["Status"] = {"select": {}}
    props["Total Validado"] = {"number": {"format": "number"}}
    props["Event ID"] = {"number": {"format": "number"}}
    props["Row Key"] = {"rich_text": {}}

    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": "Validación por Hora (Minimal)"}}],
        "properties": props
    }
    r = requests.post(f"{API_BASE}/databases", headers=_headers(token), data=json.dumps(payload))
    if not r.ok:
        raise RuntimeError(f"Error creando DB: {r.status_code} {r.text}")
    db_id = r.json()["id"]

    # alias del título
    db = get_database(token, db_id)
    title_prop = get_title_prop_name(db)
    if title_prop:
        _ADDED_ALIASES["Event Name"] = title_prop
    return db_id

def ensure_relation(token: str, db_id: str, rel_prop_name: str, target_db_id: str):
    if not rel_prop_name or not target_db_id:
        return None
    db = get_database(token, db_id)
    props = db.get("properties") or {}
    if rel_prop_name in props and props[rel_prop_name].get("type") == "relation":
        _ADDED_ALIASES[rel_prop_name] = rel_prop_name
        return rel_prop_name
    desired = {"relation": {"database_id": target_db_id, "single_property": {}}}
    r = requests.patch(f"{API_BASE}/databases/{db_id}", headers=_headers(token),
                       data=json.dumps({"properties": {rel_prop_name: desired}}))
    if not r.ok:
        raise RuntimeError(f"Error creando relación '{rel_prop_name}': {r.status_code} {r.text}")
    _ADDED_ALIASES[rel_prop_name] = rel_prop_name
    return rel_prop_name

def drop_property_if_exists(token: str, db_id: str, prop_name: str, must_be_type: str | None = None):
    """Elimina una propiedad si existe (opcionalmente sólo si coincide el tipo)."""
    db = get_database(token, db_id)
    props = db.get("properties") or {}
    for name, spec in props.items():
        if _norm(name) == _norm(prop_name):
            if must_be_type and spec.get("type") != must_be_type:
                return False
            r = requests.patch(f"{API_BASE}/databases/{db_id}", headers=_headers(token),
                               data=json.dumps({"properties": {name: None}}))
            if not r.ok:
                raise RuntimeError(f"No pude eliminar '{name}': {r.status_code} {r.text}")
            return True
    return False

def alias(name: str) -> str:
    return _ADDED_ALIASES.get(name, name)

def set_database_title(token: str, db_id: str, new_title: str):
    payload = {"title": [{"type": "text", "text": {"content": new_title}}]}
    r = requests.patch(f"{API_BASE}/databases/{db_id}", headers=_headers(token), data=json.dumps(payload))
    if not r.ok:
        raise RuntimeError(f"Error renombrando DB: {r.status_code} {r.text}")

# ---- Costos (por Event ID)
def get_title_prop_name_of_db(token: str, db_id: str) -> str:
    db = get_database(token, db_id)
    name = get_title_prop_name(db)
    if not name:
        raise RuntimeError("No pude detectar la propiedad title en la DB destino.")
    return name

def find_cost_page_by_title(token: str, target_db_id: str, title_prop: str, value) -> str | None:
    if value is None: return None
    key = f"{target_db_id}:{str(value).strip()}"
    if key in _REL_CACHE: return _REL_CACHE[key]
    payload = {"filter": {"property": title_prop, "title": {"equals": str(value).strip()}}, "page_size": 1}
    r = requests.post(f"{API_BASE}/databases/{target_db_id}/query", headers=_headers(token), data=json.dumps(payload))
    if r.ok:
        results = r.json().get("results", [])
        pid = results[0]["id"] if results else None
        _REL_CACHE[key] = pid
        return pid
    return None

# ---- Validadores (por email) con create-if-missing
def find_or_create_validator_by_email(token: str, target_db_id: str, email: str,
                                      create_if_missing: bool,
                                      preferred_email_prop: str | None = None) -> str | None:
    if not email: return None
    email = email.strip().lower()
    if email in _VALIDATOR_CACHE:
        return _VALIDATOR_CACHE[email]

    db_json = get_database(token, target_db_id)
    title_prop = get_title_prop_name(db_json)
    email_prop = preferred_email_prop or find_first_email_prop_name(db_json)

    # Query: título == email OR email_prop == email
    filt = {"or": []}
    if title_prop:
        filt["or"].append({"property": title_prop, "title": {"equals": email}})
    if email_prop:
        filt["or"].append({"property": email_prop, "email": {"equals": email}})
    if not filt["or"]:
        return None
    payload = {"filter": filt, "page_size": 1}
    r = requests.post(f"{API_BASE}/databases/{target_db_id}/query", headers=_headers(token), data=json.dumps(payload))
    if r.ok:
        results = r.json().get("results", [])
        if results:
            pid = results[0]["id"]
            _VALIDATOR_CACHE[email] = pid
            return pid

    if not create_if_missing:
        return None

    # Crear si falta: title=email, email_prop=email (si hay)
    properties = {}
    if title_prop:
        properties[title_prop] = {"title": [{"type":"text","text":{"content": email}}]}
    if email_prop:
        properties[email_prop] = {"email": email}

    r2 = requests.post(f"{API_BASE}/pages", headers=_headers(token),
                       data=json.dumps({"parent":{"database_id": target_db_id},
                                        "properties": properties}))
    if r2.ok:
        pid = r2.json()["id"]
        _VALIDATOR_CACHE[email] = pid
        return pid
    return None

# ----------------- Dataframe helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        cc = _norm(c)
        if cc in ("event name","evento","nombre evento"): mapping[c] = "Event Name"
        elif cc in ("event id","temp id","tempid","id evento","temp_id"): mapping[c] = "Event ID"
        elif cc.startswith("temp validation time") or cc in ("validation time","fecha hora","timestamp"):
            mapping[c] = "Validation Time"
        elif cc in ("status","estado"): mapping[c] = "Status"
        elif cc in ("total validado","total","validados","tickets","entradas"): mapping[c] = "Total Validado"
        elif cc in ("usuarios de validadores","temp usuarios de validadores","validador","email","validator email","usuario"):
            mapping[c] = "Usuarios de Validadores"
    return df.rename(columns=mapping) if mapping else df

def filter_df_by_costs(df: pd.DataFrame, resolver, debug=False) -> pd.DataFrame:
    ev_ids = df.get("Event ID")
    if ev_ids is None:
        return df.iloc[0:0]
    keep = []
    hits = 0
    for v in ev_ids:
        ev_id = num_from_any(v, int_out=True)
        pid = resolver(ev_id) if ev_id is not None else None
        ok = (pid is not None)
        keep.append(ok)
        if ok: hits += 1
    if debug:
        print(f"🧾 (Filtro costos) Coincidencias: {hits}/{len(df)}")
    return df.loc[keep].reset_index(drop=True)

# ----------------- Filas -----------------
def build_rows(df: pd.DataFrame, shift_hours: int = 0, debug=False,
               rel_costs_resolver=None, rel_costs_prop: str | None = None,
               rel_validators_resolver=None, rel_validators_prop: str | None = None) -> list[dict]:
    df = df.reset_index(drop=True)

    parsed_dt = df["Validation Time"].map(parse_dt_es)
    if shift_hours:
        parsed_dt = parsed_dt.map(lambda x: (x + timedelta(hours=shift_hours)) if isinstance(x, datetime) else x)

    parsed_iso_list = []
    for x in parsed_dt.astype(object):
        if isinstance(x, pd.Timestamp):
            x = x.to_pydatetime()
        if isinstance(x, datetime):
            parsed_iso_list.append(x.strftime("%Y-%m-%dT%H:%M:%S"))
        else:
            parsed_iso_list.append(None)
    parsed_iso = pd.Series(parsed_iso_list, index=df.index)

    if debug:
        total = len(df)
        ok = sum(v is not None for v in parsed_iso_list)
        bad = total - ok
        print(f"[Parse fechas] OK={ok}  BAD={bad}  (total={total})")
        if bad:
            ejemplos = df.loc[parsed_iso.isna(), "Validation Time"].dropna().unique()[:8]
            print("  Ejemplos que fallaron:", list(ejemplos))

    def email_norm(s):
        if s is None: return ""
        s = str(s).strip().lower()
        return s if s != "nan" else ""

    rows = []
    for i, r in df.iterrows():
        ev_id = num_from_any(r.get("Event ID"), int_out=True)
        iso  = parsed_iso.iat[i]
        email = email_norm(r.get("Usuarios de Validadores"))
        total_validado = num_from_any(r.get("Total Validado"), int_out=True)
        row_key = f"{ev_id or 'NA'}_{iso or 'NA'}_{email or 'NA'}"

        props = {
            alias("Event Name"): {"title":[{"type":"text","text":{"content": (r.get("Event Name") or "—")}}]},
            "Event ID": {"number": ev_id},
            "Status": {"select": {"name": str(r.get("Status"))}} if r.get("Status") else {"select": None},
            "Total Validado": {"number": total_validado},
            "Validation Time": {"date": {"start": iso}} if iso else {"date": None},
            "Row Key": {"rich_text":[{"type":"text","text":{"content": row_key}}]},
        }

        # Relación a Costos
        if rel_costs_resolver and rel_costs_prop:
            pid_cost = rel_costs_resolver(ev_id)
            props[rel_costs_prop] = {"relation": [{"id": pid_cost}]} if pid_cost else {"relation": []}

        # Relación a Validadores (por email)
        if rel_validators_resolver and rel_validators_prop:
            pid_val = rel_validators_resolver(email)
            props[rel_validators_prop] = {"relation": [{"id": pid_val}]} if pid_val else {"relation": []}

        rows.append(props)
    return rows

# ----------------- Upsert -----------------
def upsert_page(token: str, db_id: str, props: dict, sleep_s: float = 0.12):
    r = requests.post(
        f"{API_BASE}/pages",
        headers=_headers(token),
        data=json.dumps({"parent":{"database_id": db_id}, "properties": props})
    )
    if not r.ok:
        raise RuntimeError(f"Error creando page: {r.status_code} {r.text}")
    time.sleep(sleep_s)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Validación por Hora → Notion (semanal, solo relación a validadores)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--notion_token", required=True)
    ap.add_argument("--notion_parent", required=True)
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--shift_hours", type=int, default=0,
                    help="Corrimiento horario previo (p.ej. -3 si CSV está en UTC y querés AR)")
    ap.add_argument("--debug", action="store_true")

    # Relación Costos (por Event ID)
    ap.add_argument("--costs_db", default="", help="Database ID de la base de Costos/Evento para relacionar")
    ap.add_argument("--rel_prop_name", default="Costo Por Evento", help="Nombre de la propiedad de relación (Costos)")
    ap.add_argument("--only_with_costs", action="store_true",
                    help="Si se pasa (y hay --costs_db), solo sube filas cuyo Event ID exista en la DB de Costos.")

    # Relación Validadores (por email)
    ap.add_argument("--validators_db", default="", help="Database ID de 'Usuarios de Validadores'")
    ap.add_argument("--validators_rel_prop_name", default="Usuarios de Validadores",
                    help="Nombre de la propiedad de relación hacia la DB de Validadores")
    ap.add_argument("--create_validator_if_missing", action="store_true",
                    help="Si no existe el email en la DB de Validadores, crea la página automáticamente.")

    args = ap.parse_args()

    # 1) CSV
    df = load_csv_robusto(args.csv)
    df = normalize_columns(df)

    if args.debug:
        print("\n=== Preview DataFrame ===")
        print(df.head(30).to_string(index=False))
        print("Filas totales:", len(df))
        print("Columnas:", list(df.columns))
        print("Tipos de datos:", df.dtypes.to_dict())
        print("=========================\n")

    if "Validation Time" not in df.columns:
        raise SystemExit("No encuentro la columna de tiempo (Validation Time / TEMP Validation Time).")

    # 2) Crear DB NUEVA dentro de --notion_parent (con relaciones)
    db_id = ensure_db(
        args.notion_token,
        args.notion_parent,
        rel_costs_db=(args.costs_db or None),
        rel_costs_prop_name=args.rel_prop_name,
        rel_validators_db=(args.validators_db or None),
        rel_validators_prop_name=args.validators_rel_prop_name
    )
    print(f"🆕 Database creada: {db_id}")

    # 🔥 por si hubiera quedado una vieja columna email "Usuarios de Validadores": eliminarla
    try:
        removed = drop_property_if_exists(args.notion_token, db_id, "Usuarios de Validadores", must_be_type="email")
        if removed:
            print("🧹 Eliminada propiedad email 'Usuarios de Validadores' (legacy).")
    except Exception as e:
        print(f"⚠️ No pude limpiar email legacy: {e}")

    # Renombrar DB con el nombre del CSV
    csv_title = os.path.splitext(os.path.basename(args.csv))[0]
    try:
        set_database_title(args.notion_token, db_id, csv_title)
        print(f"📛 Título de la base: {csv_title}")
    except Exception as e:
        print(f"⚠️ No se pudo renombrar la base: {e}")

    # 3) Relaciones opcionales
    # --- Costos
    rel_costs_prop = None
    rel_costs_resolver = None
    if args.costs_db:
        ensure_relation(args.notion_token, db_id, args.rel_prop_name, args.costs_db)
        title_prop_costs = get_title_prop_name_of_db(args.notion_token, args.costs_db)
        def _resolver_costs(ev_id):
            if ev_id is None: return None
            return find_cost_page_by_title(args.notion_token, args.costs_db, title_prop_costs, str(ev_id))
        rel_costs_resolver = _resolver_costs
        rel_costs_prop = args.rel_prop_name

        if args.only_with_costs:
            before = len(df)
            df = filter_df_by_costs(df, rel_costs_resolver, debug=args.debug)
            after = len(df)
            print(f"🧾 Filtrado por costos: {after}/{before} filas con relación encontrada.")

    # --- Validadores
    rel_validators_prop = None
    rel_validators_resolver = None
    if args.validators_db:
        ensure_relation(args.notion_token, db_id, args.validators_rel_prop_name, args.validators_db)
        validators_db_json = get_database(args.notion_token, args.validators_db)
        validators_email_prop = find_first_email_prop_name(validators_db_json)
        def _resolver_validator(email):
            return find_or_create_validator_by_email(
                args.notion_token,
                args.validators_db,
                email,
                create_if_missing=args.create_validator_if_missing,
                preferred_email_prop=validators_email_prop
            )
        rel_validators_resolver = _resolver_validator
        rel_validators_prop = args.validators_rel_prop_name

    # 4) Construcción de filas
    rows = build_rows(
        df,
        shift_hours=args.shift_hours,
        debug=args.debug,
        rel_costs_resolver=rel_costs_resolver,
        rel_costs_prop=rel_costs_prop,
        rel_validators_resolver=rel_validators_resolver,
        rel_validators_prop=rel_validators_prop
    )

    # 5) Subida
    total = len(rows)
    for i in range(0, total, args.chunk):
        for props in rows[i:i+args.chunk]:
            upsert_page(args.notion_token, db_id, props)
        print(f"Subido: {min(i+args.chunk, total)}/{total}")
    print("✅ Listo.")

if __name__ == "__main__":
    main()
