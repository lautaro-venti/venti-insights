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
MAX_CHARS_CONV   = 9000   # deja margen; Zapier + API
REQ_TIMEOUT_SECS = 18
MAX_RETRIES      = 5
BACKOFF_BASE     = 0.7    # segundos (exponencial + jitter)
RETRY_STATUSES   = {429, 408, 500, 502, 503, 504}

# ======================= Utilidades base =======================
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
    # 1) imágenes/sistema → marcador neutro
    t = re.sub(r'\[Image\s+"[^"]*"\]', " [imagen] ", t)
    t = t.replace("[Conversation Rating Request]", " ")
    # 2) comillas “inteligentes” → ASCII
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    # 3) decode HTML y colapsa espacios
    t = html.unescape(t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    # 4) sin caracteres de control
    t = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', ' ', t)
    # 5) head+tail para preservar saludo y cierre
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
    # señales extra se agregan más abajo
    if len(kws) < 3:
        for s in ["venti","soporte","evento","entrada","app","pago","registro","qr"]:
            if s not in kws: kws.append(s)
            if len(kws) >= 5: break
    return ",".join(kws[:5])

# ======================= Reglas locales (fallback) =======================
def classify_sentiment_rule(text: str) -> str:
    t = strip_accents(text or "")
    neg_hits = ["no puedo", "no me llego", "mal servicio", "queja", "reclamo",
                "estafa", "fraude", "rechazado", "error", "fallo", "cancelado",
                "no funciona", "demora", "tarde", "reembolso", "devolucion",
                "invalido", "invalida"]
    pos_hits = ["gracias", "perfecto", "excelente", "genial", "solucionado",
                "todo ok", "buen servicio", "rapido", "funciono"]
    if any(w in t for w in neg_hits): return "Negativo"
    if any(w in t for w in pos_hits): return "Positivo"
    return "Neutro"

def classify_urgency_rule(text: str) -> str:
    t = strip_accents(text or "")
    alta = ["hoy", "ya", "urgente", "evento", "no puedo ingresar", "no puedo entrar",
            "no recibi mi entrada", "no me llego la entrada", "qr", "validador",
            "rechazado", "bloquea", "no valida", "no entra", "acceso", "no puedo comprar",
            "no puedo registrar", "no puedo crear cuenta", "registro", "crear cuenta"]
    baja = ["consulta", "informacion", "como hago", "quiero saber", "duda", "pregunta"]
    if any(w in t for w in alta): return "Alta"
    if any(w in t for w in baja): return "Baja"
    return "Media"

def csat_from_rules(sent: str, estado: str) -> int:
    s = (sent or "").lower(); e = (estado or "").lower()
    if e == "no resuelto" or e == "sin respuesta": return 1
    if e == "pendiente": return 2 if s == "negativo" else 3
    if e == "resuelto":
        return 4 if s != "negativo" else 2
    return 3

def infer_estado_final_por_resumen(resumen: str, tipo: str) -> str:
    r = resumen or ""
    t = (tipo or "").strip()
    base = strip_accents(r)
    m = re.search(r'estado\s*final\s*:\s*([a-z\s]+)', base, re.IGNORECASE)
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

# Heurística: silencio + último mensaje del agente con verbos de cierre => Resuelto
def infer_estado_por_silencio(text: str, estado_actual: str) -> str:
    t = text or ""
    # Identificar último hablante
    last_user = re.findall(r'(Usuario:|User:|Cliente:|Agente:|Agent:)', t, flags=re.IGNORECASE)
    last = last_user[-1].lower() if last_user else ""
    if "agente" not in last and "agent" not in last:
        return estado_actual  # último no fue agente
    # verbos/frases de cierre
    close_cues = [
        "listo", "hecho", "corregido", "solucionado", "resuelto", "reenviado",
        "ya pod", "probá de nuevo", "proba de nuevo", "deberías verlo",
        "deberias verlo", "liberado", "validado", "todo ok", "verificado",
        "cuenta creada", "correo corregido", "pago acreditado"
    ]
    end_segment = t[-800:].lower()
    if any(cue in strip_accents(end_segment) for cue in close_cues):
        # sin respuesta posterior del usuario (por cómo armamos el texto)
        return "Resuelto"
    return estado_actual

# ======================= Highlights =======================
def extract_highlights(text: str, k=10):
    # Cortamos por oraciones
    sents = re.split(r'(?<=[\.\!\?])\s+', (text or ""))
    # keywords ponderadas (incluye registro/correo)
    kw = [
        ("no recib", 3), ("no lleg", 3), ("qr", 3), ("valid", 2), ("rechaz", 2),
        ("pago", 2), ("devolu", 2), ("reembolso", 2), ("transfer", 2), ("swap", 2),
        ("bug", 2), ("error", 2), ("no puedo", 3), ("no funciona", 3), ("gracias", 1),
        ("solucion", 2), ("pendient", 1), ("cancel", 2), ("acceso", 2),
        ("registro", 3), ("crear cuenta", 3), ("cuenta", 2), ("correo inval", 3),
        ("mail inval", 3), ("email inval", 3), ("correo", 2), ("email", 2)
    ]
    scored = []
    for s in sents:
        ss = strip_accents(s)
        score = sum(w for kx,w in kw if kx in ss)
        score += (2 if 30 <= len(s) <= 240 else 0)  # frases densas y cortas
        if score>0:
            scored.append((score, s.strip()))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for _,s in scored[:k]]
    return (" • " + "\n • ".join(top)) if top else ""

# ======================= Fechas y rol =======================
fecha_utc_iso = ""
fecha_ba_iso  = ""
try:
    ts = int(float(created_at))
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    fecha_utc_iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    dt_ba  = dt_utc - timedelta(hours=3)  # Buenos Aires (UTC-3)
    fecha_ba_iso = dt_ba.strftime("%Y-%m-%d %H:%M:%S")
except:
    pass

rol = "customer"
try:
    if canal in ["web", "ios", "android"] and user_id and re.fullmatch(r'\d+', user_id):
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

# ======================= Prompt con few-shots =======================
asignee = authors_last_admin(authors_raw, types_raw)
link_intercom = f"https://app.intercom.com/a/apps/zgidd8i0/inbox/conversation/{conversation_id}"

FEW_SHOT_REGISTRO = """
Contexto estructurado
Tema (IC): 
Motivo (IC): 
Submotivo (IC): 
Área (IC): CX
Canal: android
Rol: customer
Fecha BA: 2025-08-25 10:00:00
Fecha UTC: 2025-08-25 13:00:00
ID: 777777
CSAT (IC): 
Asignado a: Bot
Link: https://app.intercom.com/a/apps/zgidd8i0/inbox/conversation/777777

Conversación (texto plano)
Usuario: Intento crear mi cuenta y me dice que el correo es inválido.
Agente: Revisá si el dominio tiene un typo. También podés registrarte con Google.
Usuario: El mail es juan.perez@gnail.com, ¿será por eso?
Agente: Sí, hay un error en el dominio. Probá con @gmail.com o registrate con Google. Te paso la guía.
(El usuario no responde, la conversación se cierra más tarde.)

Salida:
{"categoria":"Consulta","tipo":"Consulta","tema":"Eventos - User Ticket","motivo":"SDU (Sist. de Usuarios)","submotivo":"User no creado","urgencia":"Media","sentimiento":"Neutro","csat":4,"resumen":"android/customer: problema al crear cuenta por correo inválido (typo de dominio); ESTADO FINAL Resuelto (sin confirmación explícita); acción: orientación para corregir el mail y alternativas de registro (Google/guía).","insight":{"CRM":["Checklist de validación de mails y SSO"],"Producto":["Mensaje de error con sugerencia de dominios"],"Tech":["Validación automática de correos con typo común"]},"palabras_clave":"registro,correo,cuenta,android,typo","estado_final":"Resuelto","evidencia":"correo inválido, typo dominio, cierre sin respuesta"}
"""

FEW_SHOT_QR = """
Contexto estructurado
Tema (IC): 
Motivo (IC): 
Submotivo (IC): 
Área (IC): CX
Canal: web
Rol: customer
Fecha BA: 2025-08-22 18:04:10
Fecha UTC: 2025-08-22 21:04:10
ID: 888888
CSAT (IC): 
Asignado a: Bot
Link: https://app.intercom.com/a/apps/zgidd8i0/inbox/conversation/888888

Conversación (texto plano)
Usuario: El QR no pasa, el validador lo rechaza.
Agente: ¿Qué evento y mail? Verifico estado del QR.
Agente: Detectamos uso previo. Liberado tras verificación de identidad. Probá de nuevo.
Usuario: Ahora sí, gracias.

Salida:
{"categoria":"Reclamo","tipo":"Reclamo","tema":"Eventos - User Ticket","motivo":"QR Shield","submotivo":"Liberación QR Shield","urgencia":"Alta","sentimiento":"Positivo","csat":4,"resumen":"web/customer: QR rechazado por uso previo; ESTADO FINAL Resuelto; acción: verificación de identidad y liberación del QR.","insight":{"CRM":["Guía rápida de errores en acceso"],"Producto":["Feedback en pantalla del estado del QR"],"Tech":["Telemetría de validadores y rechazos"]},"palabras_clave":"qr,acceso,liberacion,evento,web","estado_final":"Resuelto","evidencia":"rechazo validador, liberación, agradecimiento"}
"""

# NOTA: doblamos llaves para que f-string muestre { } literales
prompt = f"""Sos analista conversacional de Venti (ticketing). Analizá una conversación CERRADA de Intercom y devolvé SOLO un JSON válido (sin backticks) con los campos pedidos, evitando textos genéricos y justificando las inferencias.

Campos y listas cerradas
- categoria ∈ {catalogo['categoria']}
- tipo ∈ {catalogo['tipo']}
- tema ∈ {catalogo['tema']}
- motivo ∈ {catalogo['motivo']}
- submotivo ∈ {catalogo['submotivo']}

Otros campos (ser específico, NO defaults):
- urgencia ∈ {{Alta, Media, Baja}} (no usar “Media” por defecto: bloquea asistir/pagar/ingresar ⇒ Alta; solo informativo ⇒ Baja)
- sentimiento ∈ {{Positivo, Neutro, Negativo}} (queja/enojo ⇒ Negativo; “gracias/solucionado” ⇒ Positivo)
- csat ∈ 1..5 (no usar 3 por defecto: Resuelto+sin queja ⇒ 4; No resuelto/Sin respuesta ⇒ 1–2)
- resumen: **2–4 oraciones**, incluir SIEMPRE: (a) canal/rol, (b) qué pidió o reportó, (c) causa o contexto concreto, (d) ESTADO FINAL {{Resuelto | No resuelto | Pendiente | Sin respuesta}}, (e) acción realizada. Evitar frases genéricas.
- insight: JSON con claves de áreas solo si aplica. Áreas: **CRM** (CX/CS/Field OPS), **Producto**, **Tech**, **Administración**. Cada área contiene 1–3 acciones concisas (máx 14 palabras) y accionables.
- palabras_clave: 3–5 en minúsculas, sin espacios, separadas por coma.
- estado_final ∈ {{Resuelto, Pendiente, No resuelto, Sin respuesta}} (obligatorio; coherente con el cierre)
- evidencia: 3–10 palabras que justifiquen urgencia/sentimiento/csat/estado_final (sin PII ni enlaces)

REGLAS:
- Si Intercom trae Tema/Motivo/Submotivo, usalos tal cual. Si faltan, elegí el mejor match del catálogo (sin inventar).
- Considerá el **inicio y cierre** de la conversación y los **highlights** para decidir urgencia/estado/csat.

EJEMPLOS (guía de estilo y detalle):
{FEW_SHOT_REGISTRO}

{FEW_SHOT_QR}

Contexto estructurado real
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

Highlights (frases relevantes)
{hl if hl else '(sin highlights)'}

Conversación (texto plano)
{conversation_for_prompt}

Formato de salida (JSON plano, sin texto extra):
{{
  "categoria": "...",
  "tipo": "...",
  "tema": "...",
  "motivo": "...",
  "submotivo": "...",
  "urgencia": "...",
  "sentimiento": "...",
  "csat": 3,
  "resumen": "...",
  "insight": {{
     "CRM": ["..."],
     "Producto": ["..."],
     "Tech": ["..."],
     "Administración": ["..."]
  }},
  "palabras_clave": "...,...,...",
  "estado_final": "...",
  "evidencia": "..."
}}"""

# ======================= Llamada a Gemini con retry/backoff =======================
if not gemini_api_key:
    raise Exception("Falta 'gemini_api_key' en los Input Data del paso Code.")

def call_gemini_with_retry(api_key: str, prompt_text: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body).encode("utf-8"), timeout=REQ_TIMEOUT_SECS)
            if resp.status_code in RETRY_STATUSES:
                last_err = f"{resp.status_code} {resp.reason}"
                sleep_s = min((BACKOFF_BASE * (2 ** (attempt-1))) * (1 + random.random()*0.25), 12)
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp.text, None
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            sleep_s = min((BACKOFF_BASE * (2 ** (attempt-1))) * (1 + random.random()*0.25), 12)
            time.sleep(sleep_s)
    return None, last_err

raw, llm_error = call_gemini_with_retry(gemini_api_key, prompt)

# ======================= Parse robusto =======================
def extract_ai_json(raw_text):
    if not raw_text:
        return {}
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
    "insight": "Registrar el caso y monitorear patrones; no hay acción clara por falta de contexto.",
    "tema_inferido": "Eventos - User Ticket",
    "motivo_inferido": "Consulta por evento",
}

categoria   = norm_match(ai.get("categoria",""), CATEGORIAS, DEFAULTS["categoria"])
tipo_norm   = norm_match(ai.get("tipo",""), TIPOS, "Consulta")
urgencia    = norm_match(ai.get("urgencia",""), URGENCIAS, DEFAULTS["urgencia"])
sentimiento = norm_match(ai.get("sentimiento",""), SENTS, DEFAULTS["sentimiento"])
resumen     = clean(ai.get("resumen")) or DEFAULTS["resumen"]
insight     = clean(ai.get("insight")) or DEFAULTS["insight"]
palabras_in = clean(ai.get("palabras_clave"))
estado_ai   = clean(ai.get("estado_final"))

def normalize_csat(value):
    try:
        v = int(str(value).strip())
        return v if 1 <= v <= 5 else None
    except:
        return None

csat_ai     = normalize_csat(ai.get("csat"))
csat_ic     = normalize_csat(csat_input_raw)
csat        = csat_ic if csat_ic is not None else csat_ai

tema_ai_inf      = clean(ai.get("tema")) or clean(ai.get("tema_inferido"))
motivo_ai_inf    = clean(ai.get("motivo")) or clean(ai.get("motivo_inferido"))
submotivo_ai_inf = clean(ai.get("submotivo")) or clean(ai.get("submotivo_inferido"))

tema_inferido      = clean(tema_ic) or tema_ai_inf or DEFAULTS["tema_inferido"]
motivo_inferido    = clean(motivo_ic) or motivo_ai_inf or DEFAULTS["motivo_inferido"]
submotivo_inferido = clean(submotivo_ic) or submotivo_ai_inf or motivo_inferido

tema_final      = tema_ic or tema_inferido
motivo_final    = motivo_ic or motivo_inferido
submotivo_final = submotivo_ic or submotivo_inferido

palabras_clave  = build_keywords(palabras_in, tema_final, motivo_final, submotivo_final, canal)
assigned_to     = authors_last_admin(authors_raw, types_raw)

# Estado final: primero IA, luego resumen, luego silencio con cues
estado_final = estado_ai if estado_ai in ESTADOS else infer_estado_final_por_resumen(resumen, tipo_norm)
estado_final = infer_estado_por_silencio(conversation_for_prompt, estado_final)

# ======================= Anti-defaults y coherencias =======================
# 1) Si vino el combo neutro/media/3 → recalcular con reglas
if (sentimiento == "Neutro" and urgencia == "Media" and (csat is None or csat == 3)):
    sentimiento = classify_sentiment_rule(conversation_for_prompt)
    urgencia    = classify_urgency_rule(conversation_for_prompt)
    csat        = csat_from_rules(sentimiento, estado_final)

# 2) Completar CSAT si quedó vacío
if csat is None:
    csat = csat_from_rules(sentimiento, estado_final)

# 3) Evitar “Resuelto + CSAT 3” salvo tono negativo explícito
if estado_final == "Resuelto" and csat == 3 and sentimiento != "Negativo":
    csat = 4

# 4) Castigo claro para no-resolución / sin respuesta
if estado_final in ("No resuelto", "Sin respuesta"):
    csat = 1

# 5) Refuerza keywords con señales
extra_tokens = []
if sentimiento == "Negativo": extra_tokens.append("queja")
if urgencia == "Alta":        extra_tokens.append("bloqueante")
if estado_final in ("No resuelto", "Pendiente"): extra_tokens.append("sinresolucion")
curr = [p for p in (palabras_clave or "").split(",") if p]
for tok in extra_tokens:
    if tok not in curr:
        curr.append(tok)
palabras_clave = ",".join(curr[:5])

# ======================= Fallbacks de Resumen / Insight (si genéricos) =======================
GENERIC_RES = {"caso sin resumen explícito. se requiere revisión.", "caso sin resumen explicito. se requiere revision.", ""}
GENERIC_INS = {
    "registrar el caso y monitorear patrones; no hay acción clara por falta de contexto.",
    "registrar el caso y monitorear patrones; no hay accion clara por falta de contexto.",
    ""
}

def synthesize_summary_rich(text, canal, rol, estado):
    t = strip_accents(text or "")
    # detectar “tema” con más granularidad
    if "registro" in t or "crear cuenta" in t or "cuenta" in t or "correo inval" in t or "email inval" in t or "mail inval" in t:
        causa = "correo inválido/typo" if ("inval" in t or "typo" in t) else "bloqueo en registro"
        return (f"El usuario intenta crear una cuenta en Venti desde {canal} ({rol}) y encuentra un error en el registro, "
                f"con indicación de {causa}. Se solicita ayuda para completar el alta y acceder a sus entradas. "
                f"Se brindan pasos de verificación/corrección del email y alternativas (SSO/guía). "
                f"ESTADO FINAL {estado}.")
    if "qr" in t or "validad" in t or "acceso" in t:
        return (f"El usuario reporta rechazo del QR/validador en acceso desde {canal} ({rol}). "
                f"Se investiga el estado del ticket y se ejecuta la acción correctiva (p.ej., liberación/verificación de identidad). "
                f"Usuario informado con pasos para reintentar. ESTADO FINAL {estado}.")
    if "no recib" in t or "no lleg" in t:
        return (f"El usuario indica que no recibió la entrada por correo (canal {canal}, rol {rol}). "
                f"Se verifica email y se gestiona reenvío/corrección de dirección. "
                f"Se informa ventana de recepción y verificación en app. ESTADO FINAL {estado}.")
    if "pago" in t or "rechaz" in t:
        return (f"El usuario consulta/incidencia de pago en {canal} ({rol}). "
                f"Se revisan causas de rechazo y se brindan pasos (medio alternativo/reintento). "
                f"Se registran evidencias para seguimiento. ESTADO FINAL {estado}.")
    # genérico enriquecido
    return (f"El usuario contacta por una gestión/consulta en {canal} ({rol}). "
            f"Se identifican posibles causas y se brindan pasos concretos de solución/seguimiento. "
            f"ESTADO FINAL {estado}.")

def synthesize_insight_rich(text):
    t = strip_accents(text or "")
    if "registro" in t or "crear cuenta" in t or "correo inval" in t or "email inval" in t or "mail inval" in t:
        return ("Revisar la validación de email en registro (detección de typos y sugerencias de dominios) e incorporar mensajes de error con correcciones guíadas. "
                "Ofrecer SSO visible (Google/Apple) y un flujo de recuperación/alta más claro.")
    if "qr" in t or "validad" in t or "acceso" in t:
        return ("Mejorar feedback del estado del QR en la app y fortalecer la telemetría de validadores; publicar guía de errores de acceso en CX y reducir el tiempo de liberación con automatismos.")
    if "no recib" in t or "no lleg" in t:
        return ("Agregar CTA de reenvío de entradas y verificador de correo en la UI; automatizar el reenvío con job idempotente y alertas si el mail rebota.")
    if "pago" in t or "rechaz" in t:
        return ("Hacer visibles las causas de rechazo de pago, guiar al usuario con alternativas y capturar métricas de PSP para priorizar fixes; reforzar guiones de CX por medio de pago.")
    return ("Establecer checklist de diagnóstico en CX, clarificar estados en la UI y ampliar trazabilidad técnica para reducir tiempos de resolución.")

if strip_accents(resumen) in GENERIC_RES:
    resumen = synthesize_summary_rich(conversation_for_prompt, canal, rol, estado_final)

if strip_accents(insight) in GENERIC_INS:
    insight = synthesize_insight_rich(conversation_for_prompt)

# Evidencia opcional
evidencia_llm = clean(ai.get("evidencia"))

# ======================= OUTPUT =======================
return {
    # IA normalizada
    "categoria": categoria,
    "tipo": tipo_norm,
    "urgencia": urgencia,
    "sentimiento": sentimiento,
    "csat": csat,
    "resumen": resumen,
    "insight": insight,
    "palabras_clave": palabras_clave,
    "estado_final": estado_final,
    "evidencia": evidencia_llm,

    # Taxonomías (crudo / inferido / final)
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

    # Metadatos
    "assigned_to": assigned_to,
    "rol": rol,
    "fecha_ba": fecha_ba_iso,
    "fecha_utc": fecha_utc_iso,
    "link_intercom": link_intercom,

    # Debug
    "llm_error": (llm_error or "")[:500],
    "raw_gemini": (raw or "")[:10000]
}
