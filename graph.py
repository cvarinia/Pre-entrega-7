"""Grafo principal del orquestador multi-agente (topologia jerarquica).

    START -> supervisor
    supervisor --(arista condicional)--> investigador | analista | aprobacion_envio | sintesis
    investigador -> supervisor      (arista fija: siempre vuelve)
    analista     -> supervisor      (arista fija: siempre vuelve)
    aprobacion_envio --(arista condicional)--> enviar_lista | sintesis     (PE7: HITL)
    enviar_lista -> sintesis
    sintesis     -> END

Los especialistas nunca se hablan entre si: todo pasa por el Supervisor.

Novedades de la PE7:
    - El grafo se compila con un CHECKPOINTER (Redis en la API, memoria en la
      CLI y los tests). Cada ejecucion usa un thread_id propio: en la API, el
      job_id. Eso es lo que permite pausar en el HITL y reanudar despues.
    - Al FINALIZAR, si la persona pidio enviar la lista y la tarea quedo
      validada, el grafo pasa por la aprobacion humana antes de enviar.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import crear_nodo_analista
from agents.research_agent import crear_nodo_investigador
from agents.supervisor import crear_nodo_supervisor
from agents.synthesis_agent import crear_nodo_sintesis
from app.hitl import crear_nodo_aprobacion, crear_nodo_envio, rutear_tras_aprobacion
from comun.config import Configuracion
from comun.llm import construir_llm
from state import TIPOS_DEL_ESTADO, EstadoOrquestador


def rutear(estado: EstadoOrquestador) -> str:
    """Arista condicional del Supervisor: devuelve el nombre del proximo nodo.

    Al FINALIZAR hay dos caminos:
      - envio pedido Y tarea validada -> aprobacion humana (HITL);
      - cualquier otro caso           -> sintesis directa.
    Una lista que no paso la validacion nunca llega a pedir aprobacion: no se
    le pregunta a una persona si quiere enviar algo que el sistema no confirmo.
    """
    if estado["next_agent"] != "FINALIZAR":
        return estado["next_agent"]  # "investigador" o "analista"
    if estado.get("solicita_envio") and estado.get("task_completed"):
        return "aprobacion_envio"
    return "sintesis"


def construir_grafo(
    config: Configuracion,
    modelo: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Arma y compila el grafo.

    Args:
        config: Configuracion de la corrida.
        modelo: Se puede inyectar un modelo simulado para probar el cableado
            sin gastar llamadas a la API (ver tests/).
        checkpointer: Donde se guarda el estado entre pasos. Sin checkpointer
            el grafo funciona, pero NO puede pausarse en el HITL.
    """
    modelo = modelo or construir_llm(config)

    constructor = StateGraph(EstadoOrquestador)
    constructor.add_node("supervisor", crear_nodo_supervisor(modelo, config))
    constructor.add_node("investigador", crear_nodo_investigador(modelo, config))
    constructor.add_node("analista", crear_nodo_analista(modelo, config))
    constructor.add_node("aprobacion_envio", crear_nodo_aprobacion())
    constructor.add_node("enviar_lista", crear_nodo_envio(config))
    constructor.add_node("sintesis", crear_nodo_sintesis(modelo, config))

    constructor.add_edge(START, "supervisor")
    constructor.add_conditional_edges(
        "supervisor", rutear, ["investigador", "analista", "aprobacion_envio", "sintesis"]
    )
    constructor.add_edge("investigador", "supervisor")
    constructor.add_edge("analista", "supervisor")
    constructor.add_conditional_edges(
        "aprobacion_envio", rutear_tras_aprobacion, ["enviar_lista", "sintesis"]
    )
    constructor.add_edge("enviar_lista", "sintesis")
    constructor.add_edge("sintesis", END)

    return constructor.compile(checkpointer=checkpointer)


def estado_inicial(pedido: str) -> dict[str, Any]:
    """Estado de arranque con todas las parcelas inicializadas."""
    return {
        "messages": [HumanMessage(content=pedido)],
        "ronda": 0,
        "instruccion_actual": "",
        "aportes": [],
        "decisiones": [],
        "task_completed": False,
        "motivo_cierre": "",
        "solicita_envio": False,
        "decision_envio": None,
        "envio": None,
    }


def config_de_ejecucion(thread_id: str, config: Configuracion) -> dict[str, Any]:
    """Config de LangGraph para una ejecucion: su hilo de checkpoints + el techo de pasos.

    El mismo thread_id se usa al arrancar y al reanudar despues de la aprobacion.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.limite_recursion_grafo,
        "run_name": "orquestador",  # nombre del span raiz en el dashboard de trazas
    }


@asynccontextmanager
async def checkpointer_redis(redis_url: str) -> AsyncIterator[BaseCheckpointSaver]:
    """Abre el checkpointer de Redis y crea sus indices la primera vez.

    Uso:
        async with checkpointer_redis(url) as saver:
            grafo = construir_grafo(config, checkpointer=saver)

    Requiere Redis 8 (o Redis Stack): el checkpointer usa los modulos de
    busqueda y JSON que vienen incluidos ahi.

    Serializador: el estado guarda modelos Pydantic propios (Aporte,
    RegistroDecision...). Por seguridad, LangGraph solo reconstruye tipos que
    se declaran como permitidos; ver serializador_redis().
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        saver.serde = serializador_redis()
        await saver.asetup()
        yield saver


def serializador_redis():
    """Serializador del checkpointer de Redis, con los tipos del estado permitidos.

    Redis guarda los checkpoints como JSON (no como msgpack, que es lo que usa
    el checkpointer en memoria). Por eso hay que permitir los tipos en las DOS
    listas: si falta `allowed_json_modules`, los modelos Pydantic del estado
    vuelven de Redis como diccionarios y el grafo falla al reanudar
    ("'dict' object has no attribute 'evidencia'").
    """
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    return JsonPlusRedisSerializer(
        allowed_json_modules=TIPOS_DEL_ESTADO,
        allowed_msgpack_modules=TIPOS_DEL_ESTADO,
    )


def checkpointer_en_memoria() -> BaseCheckpointSaver:
    """Checkpointer en memoria para la CLI y los tests (no necesita Redis).

    La logica de pausa y reanudacion es identica a la de Redis; solo cambia
    donde se guarda el estado (y que se pierde al cerrar el proceso).
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_DEL_ESTADO))
