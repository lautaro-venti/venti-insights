# src/ai_client_openai.py
import os
from typing import Dict, Any
from openai import OpenAI

_OPENAI_MODEL_WEEKLY = os.getenv("OPENAI_WEEKLY_MODEL", "gpt-4.1-mini")
_OPENAI_MODEL_AREAS  = os.getenv("OPENAI_AREAS_MODEL",  "gpt-4.1-mini")

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no seteada")
    return OpenAI(api_key=api_key)

def chat_json(messages, model=None, temperature=0.2, timeout=90) -> Dict[str, Any]:
    """
    Devuelve JSON (dict). Usa response_format=json_object para robustez.
    """
    client = get_client()
    mdl = model or _OPENAI_MODEL_WEEKLY
    resp = client.chat.completions.create(
        model=mdl,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
        timeout=timeout,
    )
    content = resp.choices[0].message.content
    import json
    return json.loads(content)

def weekly_review_prompt(week_span: str, sales_summary: dict, support_summary: dict, event_kpis: list) -> dict:
    system = (
        "Eres analista de CX/Operaciones en ticketing (Argentina). "
        "Analiza la semana con foco en ventas por evento y fricción en soporte. "
        "Devuelve JSON estrictamente ajustado al schema pedido. Sé accionable y breve."
    )
    user = {
        "week_span": week_span,  # "2025-10-20 → 2025-10-26"
        "sales_summary": sales_summary,    # dict compacto (top5, total_ars, total_tickets)
        "support_summary": support_summary,# dict compacto (totales, por área, top motivos)
        "event_kpis": event_kpis[:50],     # lista de dicts con métricas por evento
        "schema": {
            "weekly_narrative": "list[str] (3-6 bullets)",
            "highlights": "list[{title, detail, evidence:[event_name|motivo], priority:1-5}]",
            "segments": {
                "best_sellers_with_issues": "list[event_name]",
                "best_sellers_without_issues": "list[event_name]",
                "low_sellers_with_issues": "list[event_name]"
            },
            "global_actions": "list[{action, why, expected_impact, owner_suggested}]"
        }
    }
    return [{"role":"system","content":system},{"role":"user","content":str(user)}]

def area_suggestions_prompt(global_kpis: dict, area_pains: dict) -> dict:
    system = (
        "Actúas como consultor de proceso. Propón features/acciones por área "
        "(CX, Customer Success, Field OPS, Administración) priorizando CS/Field/Admin "
        "si hay poca carga. Devuelve JSON con impacto×esfuerzo y prioridad (1=alta)."
    )
    user = {
        "global_kpis": global_kpis,
        "area_pains": area_pains,
        "schema": {
            "areas": {
                "CX": "list[{feature, por_que, impacto, effort, prioridad(1-5), owner}]",
                "Customer Success": "list[...]", 
                "Field OPS": "list[...]", 
                "Administracion": "list[...]"
            }
        }
    }
    return [{"role":"system","content":system},{"role":"user","content":str(user)}]

def generate_weekly_review(week_span, sales_summary, support_summary, event_kpis) -> dict:
    msgs = weekly_review_prompt(week_span, sales_summary, support_summary, event_kpis)
    return chat_json(msgs, model=_OPENAI_MODEL_WEEKLY)

def generate_area_suggestions(global_kpis, area_pains) -> dict:
    msgs = area_suggestions_prompt(global_kpis, area_pains)
    return chat_json(msgs, model=_OPENAI_MODEL_AREAS)
