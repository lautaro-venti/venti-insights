# -*- coding: utf-8 -*-
# Venti · Intercom → JSON para Sheets/Notion (Zapier)
# + Salida estructurada (responseSchema) · prompt en parts
# + Resumen e Insight determinísticos (fallback)
# + Anti-PII · Prioridad ejecutiva
# + Auditoría (conversation_sanitizada, highlights_llm, prompt_llm_preview)
# + Guardrails (validaciones y triggers más estrictos)

import re, json, html, requests, time, random
from datetime import datetime, timezone, timedelta

# ======================= INPUTS =======================
get = lambda k: (input_data.get(k, "") or "").strip()
authors_raw      = get("authors_raw")
types_raw        = get("types_raw")
api_token        = get("api_token")
user_id          = get("user_id")
canal            = get("canal").lower()
conversation_id  = get("conversation_id")
created_at       = get("created_at")        # epoch (string o int)
bodies_raw       = input_data.get("bodies_raw", "") or ""
tema_ic          = get("tema_ic")
motivo_ic        = get("motivo_ic")
submotivo_ic     = get("submotivo_ic")
area_ic          = get("area_ic")
csat_input_raw   = get("csat_input_raw")
gemini_api_key   = get("gemini_api_key")    # OBLIGATORIO

# ======================= Constantes =======================
MAX_CHARS_CONV       = 8000
REQ_TIMEOUT_SECS     = 22
MAX_RETRIES          = 5
BACKOFF_BASE         = 0.7
RETRY_STATUSES       = {429, 408, 500, 502, 503, 504}
PROMPT_PREVIEW_MAX   = 9000
MODEL_NAME           = "gemini-1.5-flash"

# ======================= Utils =======================
def bearer_header(token: str) -> str:
    t = (token or "").strip()
    if not t: return ""
    return t if t.lower().startswith("bearer ") else f"Bearer {t}"

def clean(v):
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in {"null","none","undefined","n/a","na","-","--"} else s

def strip_accents(s: str) -> str:
    return (s or "").lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")

def sanitize_conversation(text: str) -> str:
    t = text or ""
    # proteger llaves para no romper JSON del LLM
    t = t.replace("{", "⦃").replace("}", "⦄")
    # marcas de sistema/imagen
    t = re.sub(r'\[Image\s+"[^"]*"\]', " [imagen] ", t)
    t = t.replace("[Conversation Rating Request]", " ")
    # decod HTML y remover tags
    t = html.unescape(t)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"</?p\s*/?>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"</?[^>]+>", " ", t)
    # comillas “inteligentes”
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    # normalizar puntuación repetida y espacios
    t = re.sub(r"[,\.\-_/]{3,}", " … ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    # sin caracteres de control
    t = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', ' ', t)
    # truncado head+tail
    if len(t) > MAX_CHARS_CONV:
        head = t[: int(MAX_CHARS_CONV * 0.45)]
        tail = t[-int(MAX_CHARS_CONV * 0.45):]
        t = head + "\n...\n" + tail
    return t

def authors_last_admin(authors_csv: str, types_csv: str) -> str:
    authors = [a.strip() for a in (authors_csv or "").split(",")] if authors_csv else []
    types   = [t.strip() for t in (types_csv or "").split(",")] if types_csv else []
    for a, t in zip(authors[::-1], types[::-1]):
        if (t or "").lower() == "admin":
            return a or "Admin"
    return "Bot"

def norm_match(value, allowed, default_):
    v = clean(value)
    if not v: return default_
    key = strip_accents(v)
    NORMALIZA = {
        "error tecnico":"error técnico","agradecimientos":"agradecimiento","quejas":"queja","reclamos":"reclamo",
        "sugerencias":"sugerencia","consulta":"consulta","otro":"otro",
        "alta":"alta","media":"media","baja":"baja",
        "positivo":"positivo","negativo":"negativo","neutro":"neutro",
        "informativo bot":"informativo bot","pendiente de respuesta":"pendiente de respuesta","sin respuesta":"sin respuesta"
    }
    key = NORMALIZA.get(key, key)
    for a in allowed:
        if key == strip_accents(a): return a
    return default_

def build_keywords(existing, tema, motivo, submotivo, canal):
    kws = []
    if isinstance(existing, str) and existing.strip():
        for tok in existing.split(","):
            t = tok.strip().lower().replace(" ", "")
            if t and t not in kws: kws.append(t)
    for extra in [tema, motivo, submotivo, canal]:
        e = re.sub(r"\s+", " ", (extra or "").strip().lower()).replace(" ", "")
        if e and e not in kws: kws.append(e)
    if len(kws) < 3:
        for s in ["venti","soporte","evento","entrada","app","pago","registro","qr"]:
            if s not in kws: kws.append(s)
            if len(kws) >= 5: break
    return ",".join(kws[:5])

# ======================= Reglas locales =======================
def classify_sentiment_rule(text: str) -> str:
    t = strip_accents(text or "")
    neg_hits = ["no puedo","no me llego","mal servicio","queja","reclamo","estafa","fraude","rechazado","error",
                "fallo","cancelado","no funciona","demora","tarde","reembolso","devolucion","invalido","invalida"]
    pos_hits = ["gracias","perfecto","excelente","genial","solucionado","todo ok","buen servicio","rapido","funciono"]
    if any(w in t for w in neg_hits): return "Negativo"
    if any(w in t for w in pos_hits): return "Positivo"
    return "Neutro"

def classify_urgency_rule(text: str) -> str:
    t = strip_accents(text or "")
    alta = ["hoy","ya","urgente","evento","no puedo ingresar","no puedo entrar",
            "no recibi mi entrada","no me llego la entrada","qr","validador","rechazado","bloquea",
            "no valida","no entra","acceso","no puedo comprar","no puedo registrar","no puedo crear cuenta","registro","crear cuenta"]
    baja = ["consulta","informacion","como hago","quiero saber","duda","pregunta"]
    if any(w in t for w in alta): return "Alta"
    if any(w in t for w in baja): return "Baja"
    return "Media"

def csat_from_rules(sent: str, estado: str) -> int:
    s = (sent or "").lower(); e = (estado or "").lower()
    if e in ("no resuelto","sin respuesta"): return 1
    if e == "pendiente": return 2 if s == "negativo" else 3
    if e == "resuelto": return 4 if s != "negativo" else 2
    return 3

def infer_estado_final_por_resumen(resumen: str, tipo: str) -> str:
    r = resumen or ""
    t = (tipo or "").strip()
    base = strip_accents(r)
    m = re.search(r'estado\s*final\s*[:\-]\s*([a-z\s]+)', base, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if "no resuelto" in val:   return "No resuelto"
        if "sin respuesta" in val: return "Sin respuesta"
        if "resuelto" in val:      return "Resuelto"
        if "pendiente" in val:     return "Pendiente"
    if t == "Sin respuesta":          return "Sin respuesta"
    if t == "Pendiente de respuesta": return "Pendiente"
    if any(w in base for w in ["resuelto","solucionado","enviado correctamente","reenvio correcto","corregido","listo"]):
        return "Resuelto"
    if any(w in base for w in ["no se pudo","no pudimos","rechazado","fallo","error persistente","sin poder resolver"]):
        return "No resuelto"
    if any(w in base for w in ["pendiente","aguardamos","a la espera","solicitud de datos","falta informacion","verificacion pendiente"]):
        return "Pendiente"
    return "Pendiente"

def infer_estado_por_silencio(text: str, estado_actual: str) -> str:
    t = text or ""
    last_user = re.findall(r'(Usuario:|User:|Cliente:|Agente:|Agent:)', t, flags=re.IGNORECASE)
    last = last_user[-1].lower() if last_user else ""
    if "agente" not in last and "agent" not in last:
        return estado_actual
    close_cues = ["listo","hecho","corregido","solucionado","resuelto","reenviado","ya pod","probá de nuevo","proba de nuevo",
                  "deberías verlo","deberias verlo","liberado","validado","todo ok","verificado","cuenta creada","correo corregido","pago acreditado"]
    end_segment = t[-800:].lower()
    if any(cue in strip_accents(end_segment) for cue in close_cues):
        return "Resuelto"
    return estado_actual

# ======================= Highlights =======================
def extract_highlights(text: str, k=10):
    sents = re.split(r'(?<=[\.\!\?])\s+', (text or ""))
    kw = [("no recib",3),("no lleg",3),("qr",3),("valid",2),("rechaz",2),("pago",2),("devolu",2),("reembolso",2),
          ("transfer",2),("swap",2),("bug",2),("error",2),("no puedo",3),("no funciona",3),("gracias",1),
          ("solucion",2),("pendient",1),("cancel",2),("acceso",2),("registro",3),("crear cuenta",3),("cuenta",2),
          ("correo inval",3),("mail inval",3),("email inval",3),("factur",2),("liquidac",2)]
    scored=[]
    for s in sents:
        ss=strip_accents(s)
        score=sum(w for kx,w in kw if kx in ss)
        score+=(2 if 30<=len(s)<=240 else 0)
        if score>0: scored.append((score,s.strip()))
    scored.sort(reverse=True, key=lambda x: x[0])
    top=[s for _,s in scored[:k]]
    return (" • " + "\n • ".join(top)) if top else ""

# ======================= Fechas y rol =======================
fecha_utc_iso = ""
fecha_ba_iso  = ""
try:
    ts = int(float(created_at))
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    fecha_utc_iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    dt_ba  = dt_utc - timedelta(hours=3)  # UTC-3
    fecha_ba_iso = dt_ba.strftime("%Y-%m-%d %H:%M:%S")
except:
    pass

rol = "customer"
try:
    if canal in ["web","ios","android"] and user_id and re.fullmatch(r'\d+', user_id):
        url = f"https://venti.com.ar/api/user/{user_id}"
        headers = {"Authorization": bearer_header(api_token), "Content-Type": "application/json; charset=utf-8"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            rol = (r.json().get('user', {}).get('role', '') or 'customer').lower()
        else:
            if canal == "whatsapp": rol = "admin"
            elif canal == "instagram": rol = "customer"
    else:
        if canal == "whatsapp": rol = "admin"
        elif canal == "instagram": rol = "customer"
except:
    if canal == "whatsapp": rol = "admin"

conversation_for_prompt = sanitize_conversation(bodies_raw)
hl = extract_highlights(conversation_for_prompt, k=10)

# ======================= Catálogo Venti =======================
catalogo = {
  "categoria": ["Consulta","Reclamo","Sugerencia","Queja","Agradecimiento","Error técnico","Otro"],
  "tipo": ["Consulta","Pedido","Reclamo","Duplicado","Pendiente de respuesta","Sin respuesta","Informativo Bot","Spam"],
  "tema": ["Eventos - User Ticket","Eventos - User Productora","Lead Comercial","Anuncios & Notificaciones","Duplicado","Desvío a Intercom","Sin respuesta"],
  "motivo": ["Caso Excepcional","Reenvío","Estafa por reventa","Compra externa a Venti","Consulta por evento","Team Leads & Públicas","Devolución","Pagos","Seguridad","Evento Reprogramado","Evento Cancelado","Contacto Comercial","Anuncios & Notificaciones","Duplicado","Desvío a Intercom","No recibí mi entrada","SDU (Sist. de Usuarios)","Transferencia de entradas","QR Shield","Venti Swap","Reporte","Carga masiva","Envío de invitaciones","Carga de un evento","Servicios operativos","Solicitud de reembolso","Adelantos","Liquidaciones","Estado de Cuenta","Datos de Cuenta","Altas en Venti","App de validación","Validadores","Organización de accesos en el evento","Facturación","Sin respuesta","Reclamo de usuario","Consulta sobre uso de la plataforma","Desvinculación de personal"],
  "submotivo": ["Devolución Parcial de Orden","Devolución Total de Orden","Devolución por Cobro Duplicado","Arrepentimiento no válido","Dev. no válida - Cortesía","Reporte de devoluciones","Devolución autogestiva","Solicitud de comprobante de Pago","Pago Rechazado","Compra Foránea","Pago Internacional","Medios de Pago","Facturación","Pago Dlocal","Medio de pago del adelanto","Medio de pago diferente al del contrato","Pagos diferidos","Consulta sobre saldos disponibles/utilizados","Pedido de estado de cuenta (Envios únicamente a Email)","Cambio de QR","Asignación de QRs","Cambio de QR por robo o hackeo","QRs liberados x transferencia","Liberación QR Shield","Duplicado","Notificación de reenvío","Reventa Ilegal","Reclamo por Service Charge","User no creado","Sin Respuesta","Robo y uso de tarjeta para compra de tickets","Datos no encontrados","Falla mail de verificación","Modificación de mail x typo","Reporte de ventas x provincia","Reporte de ventas x Ticket","Reporte de ventas x medio de pago","Reporte de ingresados x RRPPs","Reporte de Venta cash","Reporte de Genero de asistentes a un evento","Carga masiva de mesas","Carga masiva de RRPPs","Asignar invitaciones a RRPPs","Carga manual de RRPPs","Requerimiento RRPPs","¿Cómo vender mi entrada en Venti Swap?","¿Cómo sé si se vendió mi ticket?","¿Es gratis publicar mi ticket?","¿Cuándo se me acreditará el pago de mi entrada?","¿Puedo devolver el ticket que compré en Venti Swap?","¿Por qué no veo mis tickets a la venta en Venti Swap?","¿Qué hago si mi ticket no se vende?","¿Debo hablar con el comprador de mi ticket?","¿Cómo comprar una entrada en Venti Swap?","Validez de los tickets en Venti Swap","¿Qué hago si no me llegó el correo de verificación al mail?","¿Dónde puedo ver la entrada que compré por Venti Swap?","¿Qué pasa si se cancela el evento y compré la entrada por Venti Swap?","¿Qué pasa si se cancela el evento y vendí la entrada por Venti Swap?","¿Cómo puedo dar de baja mi publicación en Venti Swap?","Consulta por evento sin swap activo","Modificar los datos bancarios o el precio de mi entrada en Venti Swap","Consulta por evento reprogramado","Consulta sobre evento cancelado","Consulta sobre info del evento","Cancelación/Reprogramación de un evento","Crear un evento de 0","Crear un ticket nuevo","Eliminar un evento","Uso de la plataforma","Consulta para comenzar a operar con Venti","Consulta para operar con Venti","Solicitud de alta en Venti","Derivación de productora por parte de otro productor de Venti","Derivación al SDU","Derivación a Jotform","Pedido de Cambio de Datos","Solicitud de envío de mailing","Solicitud de adelantos","Solicitud de carga de pixel","Centro de ayuda","Desvinculación de personal","Consulta sobre el status del pago del evento","Consulta sobre liquidación del finde","Priorización de liquidación del finde","Envío de liquidación a un mail diferente","App","Usuarios con acceso","Reportes","Consulta sobre usuario de la app","Consulta sobre ingreso a la app de validación","Consulta sobre manejo de la app","Consulta sobre cómo validar diferentes sectores en la app","Pedido de validadores para un evento","Consulta sobre el sistema de validadores","Consulta sobre presupuesto de validadores","Consulta operativa de acceso en el evento","Envío de información sobre los accesos en el evento","Cómo comprar entradas","Ticket COMBO","Promociones y descuentos","Consulta sobre cómo vender con Venti","Modificar URL","Derivar a CX","Caso Excepcional","Reenvío","Estafa por reventa","Compra externa a Venti","Consulta por evento","Team Leads & Publicas","Devolución","Pagos","Seguridad","Evento Reprogramado","Evento Cancelado","Contacto Comercial","Anuncios & Notificaciones","Duplicado","Desvío a Intercom","No recibí mi entrada","SDU (Sist. de Usuarios)","Transferencia de entradas","QR Shield","Venti Swap","Reporte","Carga masiva","Envío de invitaciones","Carga de un evento","Servicios operativos","Solicitud de reembolso","Adelantos","Liquidaciones","Estado de Cuenta","Datos de Cuenta","Altas en Venti","App de validacion","Validadores","Organización de accesos en el evento","Facturación","Sin respuesta","Reclamo de usuario","Consulta sobre uso de la plataforma","Desvinculación de personal"]
}

# ======================= JSON Schema (Salida Estructurada) =======================
JSON_SCHEMA = {
  "type": "object",
  "required": ["categoria","tipo","tema","motivo","submotivo","urgencia","sentimiento",
               "csat","resumen","insight","palabras_clave","estado_final","evidencia"],
  "properties": {
    "categoria":  {"type": "string", "enum": catalogo["categoria"] + ["—"]},
    "tipo":       {"type": "string", "enum": list(set(catalogo["tipo"])) + ["—"]},
    "tema":       {"type": "string", "enum": catalogo["tema"] + ["—"]},
    "motivo":     {"type": "string", "enum": catalogo["motivo"] + ["—"]},
    "submotivo":  {"type": "string", "enum": catalogo["submotivo"] + ["—"]},
    "urgencia":   {"type": "string", "enum": ["Alta","Media","Baja","—"]},
    "sentimiento":{"type": "string", "enum": ["Positivo","Neutro","Negativo","—"]},
    "csat":       {"type": "integer", "minimum": 1, "maximum": 5},
    "resumen":    {"type": "string", "minLength": 30, "maxLength": 600},
    "insight":    {"type": "string", "minLength": 30, "maxLength": 600},
    "palabras_clave": {"type": "string", "pattern": r"^[a-z0-9]+(,[a-z0-9]+){2,4}$"},
    "estado_final": {"type": "string", "enum": ["Resuelto","Pendiente","No resuelto","Sin respuesta","—"]},
    "evidencia":  {"type": "string", "minLength": 10, "maxLength": 200}
  },
  "additionalProperties": False
}

# ======================= Few-shots =======================
asignee = authors_last_admin(authors_raw, types_raw)
link_intercom = f"https://app.intercom.com/a/apps/zgidd8i0/inbox/conversation/{conversation_id}"

FEW_SHOT_REGISTRO = """
[FEW-SHOT]
Contexto estructurado
Canal: android
Rol: customer
Conversación (texto plano)
Usuario: Intento crear mi cuenta y me dice que el correo es inválido.
Agente: Revisá si el dominio tiene un typo. También podés registrarte con Google.
Usuario: El mail es juan.perez@gnail.com, ¿será por eso?
Agente: Sí, hay un error en el dominio. Probá con @gmail.com o registrate con Google.
Salida:
{"categoria":"Consulta","tipo":"Consulta","tema":"Eventos - User Ticket","motivo":"SDU (Sist. de Usuarios)","submotivo":"User no creado","urgencia":"Media","sentimiento":"Neutro","csat":4,"resumen":"android/customer: problema al crear cuenta por correo inválido (typo de dominio); ESTADO FINAL Resuelto (sin confirmación explícita); acción: orientación para corregir el mail y alternativas de registro (Google/guía).","insight":"Priorizar corrección de typos en registro con sugerencias de dominios y SSO destacado; agregar texto de error accionable y link a guía. — Prioridad: Media; Métrica: tasa de altas fallidas; Tiempo: 2 semanas","palabras_clave":"registro,correo,cuenta,android,typo","estado_final":"Resuelto","evidencia":"correo inválido, typo dominio, cierre sin respuesta"}
"""

FEW_SHOT_QR = """
[FEW-SHOT]
Contexto estructurado
Canal: web
Rol: customer
Conversación (texto plano)
Usuario: El QR no pasa, el validador lo rechaza.
Agente: Detectamos uso previo. Liberado tras verificación de identidad. Probá de nuevo.
Usuario: Ahora sí, gracias.
Salida:
{"categoria":"Reclamo","tipo":"Reclamo","tema":"Eventos - User Ticket","motivo":"QR Shield","submotivo":"Liberación QR Shield","urgencia":"Alta","sentimiento":"Positivo","csat":4,"resumen":"web/customer: QR rechazado por uso previo; ESTADO FINAL Resuelto; acción: verificación de identidad y liberación del QR.","insight":"Mostrar estado del QR en la UI con causa y next-steps, y reforzar telemetría de validadores para detectar reusos. — Prioridad: Alta; Métrica: rechazos por QR; Tiempo: 1 semana","palabras_clave":"qr,acceso,liberacion,evento,web","estado_final":"Resuelto","evidencia":"rechazo validador, liberación, agradecimiento"}
"""

# ======================= Prompt (en parts) =======================
PROMPT_HEADER_Y_REGLAS = f"""Eres analista conversacional de Venti (ticketing).
Analiza una conversación CERRADA y devuelve ÚNICAMENTE un objeto JSON válido (sin backticks), con los campos pedidos.
NO inventes datos que no estén explícitos en la conversación o en el contexto. Si un dato falta, escribe "—".

Campos y listas cerradas
- categoria ∈ {catalogo['categoria']}
- tipo ∈ {catalogo['tipo']}
- tema ∈ {catalogo['tema']}
- motivo ∈ {catalogo['motivo']}
- submotivo ∈ {catalogo['submotivo']}

Otros campos (ser específico, NO defaults):
- urgencia ∈ {{Alta, Media, Baja}} (bloquea asistir/pagar/ingresar ⇒ Alta; solo informativo ⇒ Baja)
- sentimiento ∈ {{Positivo, Neutro, Negativo}} (queja/enojo ⇒ Negativo; “gracias/solucionado” ⇒ Positivo)
- csat ∈ 1..5 (Resuelto+sin queja ⇒ 4; No resuelto/Sin respuesta ⇒ 1–2)
- resumen: 2–4 oraciones e INCLUIR SIEMPRE: (a) canal/rol, (b) qué pidió o reportó, (c) causa/cońtexto concreto, (d) ESTADO FINAL {{Resuelto | No resuelto | Pendiente | Sin respuesta}}, (e) acción realizada.
- insight: 1 línea (1–2 oraciones) SIN áreas; acciones concretas y encadenadas + “— Prioridad: X; Métrica: Y; Tiempo: Z”.
- palabras_clave: 3–5 en minúsculas, sin espacios, separadas por coma.
- estado_final ∈ {{Resuelto, Pendiente, No resuelto, Sin respuesta}}
- evidencia: 3–10 palabras que justifiquen urgencia/sentimiento/csat/estado_final (sin PII ni enlaces)

Reglas de foco:
- Usa SOLO lo que está entre los delimitadores de CONTEXTO/HIGHLIGHTS/CONVERSACIÓN.
- Devuelve únicamente JSON (nada de explicaciones).

Ejemplos de estilo:
{FEW_SHOT_REGISTRO}

{FEW_SHOT_QR}
"""

CONTEXTO_Y_HIGHLIGHTS = f"""=== CONTEXTO (BEGIN) ===
Tema (IC): {tema_ic}
Motivo (IC): {motivo_ic}
Submotivo (IC): {submotivo_ic}
Área (IC): {area_ic}
Canal: {canal}
Rol: {rol}
Fecha BA: {fecha_ba_iso}
Fecha UTC: {fecha_utc_iso}
ID: {conversation_id}
CSAT (IC): {csat_input_raw}
Asignado a: {asignee}
Link: {link_intercom}
=== CONTEXTO (END) ===

=== HIGHLIGHTS (BEGIN) ===
{hl if hl else '(sin highlights)'}
=== HIGHLIGHTS (END) ===
"""

CONVERSACION_SANITIZADA = f"""=== CONVERSACIÓN (BEGIN) ===
{conversation_for_prompt}
=== CONVERSACIÓN (END) ===
"""

# ======================= Gemini (con schema) =======================
if not gemini_api_key:
    raise Exception("Falta 'gemini_api_key' en los Input Data del paso Code.")

def call_gemini_with_retry(api_key: str, parts_list: list):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # menos variabilidad en IG/WA
    temperature = 0.05 if canal in ("instagram", "whatsapp") else 0.1

    body = {
      "contents": [{
         "role": "user",
         "parts": [{"text": p} for p in parts_list]
      }],
      "generationConfig": {
        "temperature": temperature,
        "topK": 1,
        "topP": 0.9,
        "maxOutputTokens": 900,
        "responseMimeType": "application/json",
        "responseSchema": JSON_SCHEMA
        }
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body).encode("utf-8"), timeout=REQ_TIMEOUT_SECS)
            if resp.status_code in RETRY_STATUSES:
                last_err = f"{resp.status_code} {resp.reason}"
                sleep_s = min((BACKOFF_BASE * (2 ** (attempt-1))) * (1 + random.random()*0.25), 12)
                time.sleep(sleep_s); continue
            resp.raise_for_status()
            return resp.text, None
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            sleep_s = min((BACKOFF_BASE * (2 ** (attempt-1))) * (1 + random.random()*0.25), 12)
            time.sleep(sleep_s)
    return None, last_err

raw, llm_error = call_gemini_with_retry(
    gemini_api_key,
    [PROMPT_HEADER_Y_REGLAS, CONTEXTO_Y_HIGHLIGHTS, CONVERSACION_SANITIZADA]
)

# ======================= Parse robusto =======================
def extract_ai_json(raw_text):
    if not raw_text: return {}
    try:
        outer = json.loads(raw_text)
        if isinstance(outer, dict) and ("categoria" in outer or "tipo" in outer):
            return outer
        text = outer["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text = raw_text if isinstance(raw_text, str) else ""
    if not isinstance(text, str):
        return {}
    m = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL) or re.search(r'{.*}', text, re.DOTALL)
    if not m:
        try:
            return json.loads(text)
        except Exception:
            return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        s = m.group(0).strip()
        last = s.rfind("}")
        if last != -1:
            try: return json.loads(s[:last+1])
            except: return {}
        return {}

ai = extract_ai_json(raw) if raw else {}

# ======================= Normalizaciones y fusiones =======================
CATEGORIAS = set(catalogo["categoria"])
URGENCIAS  = {"Alta","Media","Baja"}
SENTS      = {"Positivo","Neutro","Negativo"}
TIPOS      = set(catalogo["tipo"])
ESTADOS    = {"Resuelto","Pendiente","No resuelto","Sin respuesta"}

DEFAULTS = {
    "categoria": "Consulta",
    "urgencia": "Media",
    "sentimiento": "Neutro",
    "resumen": "Caso sin resumen explícito. Se requiere revisión.",
    "insight": "Optimizar guía y feedback en UI, más automatismos y telemetría; medir impacto en TTR/CSAT."
}

categoria   = norm_match(ai.get("categoria",""), CATEGORIAS, DEFAULTS["categoria"])
tipo_norm   = norm_match(ai.get("tipo",""), TIPOS, "Consulta")
urgencia    = norm_match(ai.get("urgencia",""), URGENCIAS, DEFAULTS["urgencia"])
sentimiento = norm_match(ai.get("sentimiento",""), SENTS, DEFAULTS["sentimiento"])
resumen     = clean(ai.get("resumen")) or DEFAULTS["resumen"]
palabras_in = clean(ai.get("palabras_clave"))
estado_ai   = clean(ai.get("estado_final"))

def normalize_csat(value):
    try:
        v = int(str(value).strip());  return v if 1 <= v <= 5 else None
    except:  return None

csat_ai     = normalize_csat(ai.get("csat"))
csat_ic     = normalize_csat(csat_input_raw)
csat        = csat_ic if csat_ic is not None else csat_ai

tema_ai_inf      = clean(ai.get("tema")) or clean(ai.get("tema_inferido"))
motivo_ai_inf    = clean(ai.get("motivo")) or clean(ai.get("motivo_inferido"))
submotivo_ai_inf = clean(ai.get("submotivo")) or clean(ai.get("submotivo_inferido"))

tema_inferido      = clean(tema_ic) or tema_ai_inf or "Eventos - User Ticket"
motivo_inferido    = clean(motivo_ic) or motivo_ai_inf or "Consulta por evento"
submotivo_inferido = clean(submotivo_ic) or submotivo_ai_inf or motivo_inferido

tema_final      = tema_ic or tema_inferido
motivo_final    = motivo_ic or motivo_inferido
submotivo_final = submotivo_ic or submotivo_inferido

palabras_clave  = build_keywords(palabras_in, tema_final, motivo_final, submotivo_final, canal)

# Estado final: IA → resumen → silencio con cues
estado_final = estado_ai if estado_ai in ESTADOS else infer_estado_final_por_resumen(resumen, tipo_norm)
estado_final = infer_estado_por_silencio(conversation_for_prompt, estado_final)

# ======================= Anti-defaults y coherencias =======================
if (sentimiento == "Neutro" and urgencia == "Media" and (csat is None or csat == 3)):
    sentimiento = classify_sentiment_rule(conversation_for_prompt)
    urgencia    = classify_urgency_rule(conversation_for_prompt)
    csat        = csat_from_rules(sentimiento, estado_final)

if csat is None: csat = csat_from_rules(sentimiento, estado_final)
if estado_final == "Resuelto" and csat == 3 and sentimiento != "Negativo": csat = 4
if estado_final in ("No resuelto","Sin respuesta"): csat = 1

extra_tokens=[]
if sentimiento=="Negativo": extra_tokens.append("queja")
if urgencia=="Alta":        extra_tokens.append("bloqueante")
if estado_final in ("No resuelto","Pendiente"): extra_tokens.append("sinresolucion")
curr=[p for p in (palabras_clave or "").split(",") if p]
for tok in extra_tokens:
    if tok not in curr: curr.append(tok)
palabras_clave=",".join(curr[:5])

# ======================= RESUMEN determinístico (fallback) =======================
def detect_causa(text: str):
    t = strip_accents(text or "")
    if any(k in t for k in ["correo inval","email inval","mail inval","typo","registro","crear cuenta"]):
        return "registro / correo inválido"
    if any(k in t for k in ["qr","validad","rechaz","uso previo","no pasa"]):
        return "qr rechazado / uso previo"
    if any(k in t for k in ["pago","rechazad","checkout","tarjeta","mp "]):
        return "pago rechazado / checkout"
    if any(k in t for k in ["no recib","no lleg","reenvi","link de entrada"]):
        return "no recibió entradas"
    if any(k in t for k in ["reembol","devolu","refund"]):
        return "solicitud de reembolso"
    if any(k in t for k in ["factur","liquidac","estado de cuenta"]):
        return "consulta administrativa"
    return "—"

def detect_accion(text: str):
    t = strip_accents(text or "")
    if any(k in t for k in ["liberad","desbloque","validado"]): return "liberación/validación"
    if any(k in t for k in ["reenv","reenviar"]): return "reenvío de entradas"
    if any(k in t for k in ["verific","identidad"]): return "verificación de identidad"
    if any(k in t for k in ["guia","paso","instrucc"]): return "envío de guía/pasos"
    if any(k in t for k in ["solicit","datos","comprobante"]): return "solicitud de datos"
    if any(k in t for k in ["correg","typo","dominio"]): return "corrección de mail"
    if any(k in t for k in ["reembolso","refund"]): return "inicio de reembolso"
    return "—"

def resumen_has_all_requirements(s: str) -> bool:
    s2 = strip_accents(s or "")
    reqs = [
        ("canal/rol", any(k in s2 for k in [strip_accents(canal), strip_accents(rol)])),
        ("pedido", any(k in s2 for k in ["pide","reporta","indica","consulta","no recibio","no puedo","rechaz"])),
        ("causa", any(k in s2 for k in ["qr","pago","registro","correo","reembolso","entrada"])),
        ("estado", "estado final" in s2),
        ("accion", any(k in s2 for k in ["accion","reenvio","verificacion","liberacion","solicitud de datos","guia","validacion","correccion"]))
    ]
    return all(flag for _, flag in reqs)

def build_resumen_deterministico(canal, rol, text, estado, accion, causa):
    canal_rol = f"{canal}/{rol}"
    pedido = "el usuario reporta un problema" if causa == "—" else f"el usuario reporta {causa}"
    accion_txt = accion if accion != "—" else "asistencia y pasos concretos"
    return f"{canal_rol}: {pedido}. ESTADO FINAL {estado}; acción: {accion_txt}."

# ======================= Insight determinístico (fallback) =======================

def synthesize_insight_rich_line(text: str, motivo: str, submotivo: str) -> str:
    t = strip_accents(text or "")
    ideas = []
    prior = "Media"; metric = None; timebox = None

    if any(k in t for k in ["registro","crear cuenta","correo inval","email inval","mail inval"]) or "SDU" in (motivo or ""):
        ideas.append("Corregir typos en registro con sugerencia de dominios y SSO visible; mostrar error accionable con link a guía")
        prior = "Media"; metric = metric or "tasa de altas fallidas"; timebox = timebox or "2 semanas"
    if any(k in t for k in ["qr","validad","acceso","rechaz"]) or "QR Shield" in (motivo or ""):
        ideas.append("Exponer estado del QR en UI con causa y next-steps; telemetría de validadores y liberación automática tras verificación")
        prior = "Alta"; metric = metric or "rechazos por QR"; timebox = timebox or "1 semana"
    if any(k in t for k in ["pago","rechazado","reembol","devolu"]) or ("Pagos" in (motivo or "") or "Devolución" in (motivo or "")):
        ideas.append("Mostrar causa de rechazo en checkout y guiar reintento; orquestar reembolsos vía webhook con estados y notificaciones")
        prior = "Alta"; metric = metric or "TTR reembolsos p50/p90"; timebox = timebox or "1–2 semanas"
    if any(k in t for k in ["no recib","no lleg","reenvio","reenviar"]) or "No recibí mi entrada" in (motivo or ""):
        ideas.append("Agregar CTA de reenvío de entradas y verificador de rebote; job idempotente y alertas")
        metric = metric or "tasa de reenvíos exitosos"; timebox = timebox or "1 semana"

    if not ideas:
        ideas.append("Optimizar feedback en UI y sumar automatismos/telemetría para reducir fricción y TTR")
        metric = metric or "TTR p50/p90 y CSAT"; prior = "Media"; timebox = timebox or "2–4 semanas"

    core = "; ".join(ideas)
    tail = f" — Prioridad: {prior}; Métrica: {metric}; Tiempo: {timebox}"
    return re.sub(r"\s*\n+\s*", " ", (core + tail)).strip()[:600]

GENERIC_INSIGHT_RX = re.compile(
    r"\b(mejorar|optimizar)\b.{0,30}\b(experiencia|ux|ui|gu[ií]a)\b",
    re.IGNORECASE
)

insight = clean(ai.get("insight")) or ""
etq = sum(x in (insight or "") for x in ("Prioridad:", "Métrica:", "Tiempo:"))
if (len(insight) < 30) or (GENERIC_INSIGHT_RX.search(insight or "") is not None) or (etq < 2):
    insight = synthesize_insight_rich_line(conversation_for_prompt, motivo_final, submotivo_final)
else:
    insight = re.sub(r"\s*\n+\s*", " ", insight).strip()[:600]

# ======================= Prioridad ejecutiva =======================
def make_prioridad_ejecutiva(urg: str, text: str, motivo: str, submotivo: str, estado: str) -> str:
    t = strip_accents(text or "")
    step = "Diagnóstico y respuesta"
    sla  = "≤ 24 h" if urg == "Media" else ("≤ 72 h" if urg == "Baja" else "≤ 60 min")
    if any(k in t for k in ["qr","validad","acceso"]) or "QR Shield" in (motivo or ""):
        step = "Verificar identidad y liberar QR (si corresponde)"; sla  = "≤ 15 min"
    elif any(k in t for k in ["pago rechaz","pago","checkout"]) or "Pagos" in (motivo or ""):
        step = "Identificar causa de rechazo y guiar reintento/medio alternativo"; sla  = "≤ 2 h"
    elif any(k in t for k in ["reembol","devolu"]) or "Devolución" in (motivo or ""):
        step = "Disparar reembolso vía webhook y notificar estado"; sla  = "≤ 48 h"
    elif any(k in t for k in ["no recib","no lleg","reenvi"]) or "No recibí mi entrada" in (motivo or ""):
        step = "Reenviar entradas y verificar rebote/spam"; sla  = "≤ 4 h"
    elif any(k in t for k in ["registro","crear cuenta","correo inval"]) or "SDU" in (motivo or ""):
        step = "Corregir email/SSO y completar alta"; sla  = "≤ 24 h"
    return f"Prioridad Ejecutiva: {urg} — {step}; SLA objetivo {sla}; Estado: {estado}."[:200]

# ======================= Ensamblado resumen/insight con guardrails =======================
causa_det  = detect_causa(conversation_for_prompt)
accion_det = detect_accion(conversation_for_prompt)

# 1) Resumen: si vacío, corto o sin “ESTADO FINAL” ⇒ determinístico
def resumen_core_ok(s: str) -> bool:
    s2 = strip_accents(s or "")
    reqs = [
        any(k in s2 for k in ["pide","reporta","indica","consulta","no recibio","no puedo","rechaz"]),
        any(k in s2 for k in ["qr","pago","registro","correo","reembolso","entrada"]),   # causa
        any(k in s2 for k in ["accion","reenvio","verificacion","liberacion","datos","guia","validacion","correccion"])
    ]
    return all(reqs)

if (not resumen) or (len(resumen) < 30) or (not resumen_core_ok(resumen)):
    resumen = build_resumen_deterministico(canal, rol, conversation_for_prompt, estado_final, accion_det, causa_det)

# 2) Insight: si vacío/corto/genérico o sin etiquetas ⇒ determinístico
insight = clean(ai.get("insight")) or ""
if (len(insight) < 40) or GENERIC_INSIGHT_RX.search(insight or "") is not None \
   or ("Prioridad:" not in insight) or ("Métrica:" not in insight) or ("Tiempo:" not in insight):
    insight = synthesize_insight_rich_line(conversation_for_prompt, motivo_final, submotivo_final)
else:
    insight = re.sub(r"\s*\n+\s*", " ", insight).strip()[:600]

prioridad_ejecutiva = make_prioridad_ejecutiva(urgencia, conversation_for_prompt, motivo_final, submotivo_final, estado_final)

# Resumen compacto 2–4 oraciones
def enforce_summary_length(s: str, min_sent=2, max_sent=4) -> str:
    toks = re.split(r'(?<=[\.\!\?])\s+', (s or "").strip())
    toks = [t.strip() for t in toks if t.strip()]
    toks = toks[:max_sent]
    if len(toks) < min_sent and toks:
        toks = [(" ".join(toks)).strip(".") + "."]
    return " ".join(toks)

resumen = enforce_summary_length(resumen, 2, 4)

# ======================= Anti-PII =======================
PII_RX = [
    (re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'), '[mail]'),
    (re.compile(r'\b\+?\d[\d\-\s]{6,}\b'), '[tel]'),
]
def scrub_pii(s: str) -> str:
    out = s or ""
    for rx, sub in PII_RX:
        out = rx.sub(sub, out)
    return out

evidencia_llm = clean(ai.get("evidencia"))
resumen = scrub_pii(resumen)
insight = scrub_pii(insight)
evidencia_llm = scrub_pii(evidencia_llm)

# ======================= Auditoría =======================
prompt_llm_preview = (PROMPT_HEADER_Y_REGLAS + "\n\n" + CONTEXTO_Y_HIGHLIGHTS + "\n\n" + CONVERSACION_SANITIZADA)[:PROMPT_PREVIEW_MAX]

# ======================= OUTPUT =======================
return {
    "categoria": categoria,
    "tipo": tipo_norm,
    "urgencia": urgencia,
    "sentimiento": sentimiento,
    "csat": csat,
    "resumen": resumen,
    "insight": insight,
    "prioridad_ejecutiva": prioridad_ejecutiva,
    "palabras_clave": palabras_clave,
    "estado_final": estado_final,
    "evidencia": evidencia_llm,

    "tema_ic": tema_ic,
    "motivo_ic": motivo_ic,
    "submotivo_ic": submotivo_ic,
    "area_ic": area_ic,
    "tema_inferido": tema_inferido,
    "motivo_inferido": motivo_inferido,
    "submotivo_inferido": submotivo_inferido,
    "tema_final": tema_final,
    "motivo_final": motivo_final,
    "submotivo_final": submotivo_final,

    "assigned_to": authors_last_admin(authors_raw, types_raw),
    "rol": rol,
    "fecha_ba": fecha_ba_iso,
    "fecha_utc": fecha_utc_iso,
    "link_intercom": link_intercom,

    # Auditoría LLM
    "conversation_sanitizada": conversation_for_prompt,
    "highlights_llm": hl,
    "prompt_llm_preview": prompt_llm_preview,

    # Debug
    "llm_error": (llm_error or "")[:500],
    "raw_gemini": (raw or "")[:10000]
}
