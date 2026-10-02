"""Pruebas del human-in-the-loop SIN consumir la API (modelo simulado).

Usan un checkpointer EN MEMORIA (checkpointer_en_memoria): la logica de pausa y
reanudacion es exactamente la misma que con Redis, pero no hace falta tener
Redis levantado. Lo unico que cambia en produccion es DONDE se guarda el estado.

Escenarios:
    1. Envio pedido + aprobado  -> pausa, reanuda, envia, sintetiza.
    2. Envio pedido + rechazado -> pausa, reanuda, NO envia, sintetiza.
    3. Envio NO pedido          -> no hay pausa (asi corre la prueba de carga).
    4. Envio pedido + tarea sin validar -> no hay pausa ni envio.
    5. Respuesta humana invalida -> falla la validacion de Pydantic.
    6. Pausa y reanudacion con el SERIALIZADOR DE REDIS (JSON): los modelos
       del estado tienen que volver como modelos, no como diccionarios.

Uso:  python -m tests.test_hitl_offline
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.hitl import ACCION_CRITICA, pedido_pendiente  # noqa: E402
from comun.config import Configuracion  # noqa: E402
from comun.modelo_simulado import ModeloSimulado  # noqa: E402
from graph import (  # noqa: E402
    checkpointer_en_memoria,
    config_de_ejecucion,
    construir_grafo,
    estado_inicial,
    serializador_redis,
)  # noqa: E402
from state import DecisionSupervisor  # noqa: E402
from tests.test_orquestador_offline import analisis, investigacion  # noqa: E402

IDS = ["r008", "r009", "r018"]
PEDIDO = "3 cenas vegetarianas para 6 y mandale la lista de compras al almacen"


def config_de_prueba() -> Configuracion:
    # Los envios simulados de los tests van a una carpeta aparte.
    base = Configuracion.para_pruebas()
    return replace(base, ruta_logs=base.ruta_logs / "tests")


def guion(solicita_envio: bool) -> ModeloSimulado:
    """Flujo feliz: investigador -> analista -> FINALIZAR."""
    return ModeloSimulado(
        respuestas=[*investigacion(IDS), *analisis(IDS), AIMessage(content="Respuesta final simulada.")],
        decisiones=[
            DecisionSupervisor(siguiente="investigador", instruccion="Busca 3 cenas vegetarianas.", evaluacion="-"),
            DecisionSupervisor(siguiente="analista", instruccion="Calcula calorias y compras para 6.", evaluacion="-"),
            DecisionSupervisor(siguiente="FINALIZAR", evaluacion="-", solicita_envio=solicita_envio),
        ],
    )


async def arrancar(modelo: ModeloSimulado, hilo: str, config: Configuracion | None = None, checkpointer=None):
    config = config or config_de_prueba()
    grafo = construir_grafo(config, modelo=modelo, checkpointer=checkpointer or checkpointer_en_memoria())
    cfg = config_de_ejecucion(hilo, config)
    resultado = await grafo.ainvoke(estado_inicial(PEDIDO), cfg)
    return grafo, cfg, resultado


async def prueba_aprobado() -> None:
    grafo, cfg, resultado = await arrancar(guion(solicita_envio=True), "hilo-aprobado")

    # 1) El grafo se detuvo antes de enviar.
    pedido = pedido_pendiente(resultado)
    assert pedido is not None and pedido["accion"] == ACCION_CRITICA, resultado.keys()
    assert resultado.get("envio") is None
    snapshot = await grafo.aget_state(cfg)
    assert snapshot.next == ("aprobacion_envio",), snapshot.next

    # 2) Se reanuda con la respuesta humana y el MISMO thread_id.
    final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "carla"}), cfg)
    assert pedido_pendiente(final) is None
    assert final["envio"] is not None and final["decision_envio"].aprobado
    assert Path(config_de_prueba().ruta_logs.parent / final["envio"].archivo).exists()
    assert final["messages"][-1].name == "sintesis"
    print(f"OK  aprobado: pausa -> reanuda -> envio {final['envio'].envio_id} -> sintesis")


async def prueba_rechazado() -> None:
    grafo, cfg, resultado = await arrancar(guion(solicita_envio=True), "hilo-rechazado")
    assert pedido_pendiente(resultado) is not None

    final = await grafo.ainvoke(
        Command(resume={"aprobado": False, "revisor": "carla", "comentario": "Falta revisar el presupuesto"}), cfg
    )
    assert final["envio"] is None
    assert final["decision_envio"].aprobado is False
    assert final["messages"][-1].name == "sintesis"
    print("OK  rechazado: pausa -> reanuda -> sin envio -> sintesis")


async def prueba_sin_pedido_de_envio() -> None:
    _, _, resultado = await arrancar(guion(solicita_envio=False), "hilo-sin-envio")
    assert pedido_pendiente(resultado) is None
    assert resultado["envio"] is None and resultado["messages"][-1].name == "sintesis"
    print("OK  sin pedido de envio: corre de punta a punta sin pausa")


async def prueba_tarea_no_validada() -> None:
    """Pide envio, pero el analista proceso otras recetas: no se pregunta ni se envia."""
    modelo = ModeloSimulado(
        respuestas=[*investigacion(IDS), *analisis(["r008", "r009", "r001"]), AIMessage(content="Final.")],
        decisiones=[
            DecisionSupervisor(siguiente="investigador", instruccion="Busca.", evaluacion="-"),
            DecisionSupervisor(siguiente="analista", instruccion="Calcula.", evaluacion="-"),
            DecisionSupervisor(siguiente="FINALIZAR", evaluacion="-", solicita_envio=True),
        ],
    )
    _, _, resultado = await arrancar(modelo, "hilo-no-validada")
    assert resultado["task_completed"] is False
    assert pedido_pendiente(resultado) is None and resultado["envio"] is None
    print("OK  tarea no validada: no se pide aprobacion ni se envia")


async def prueba_respuesta_invalida() -> None:
    grafo, cfg, _ = await arrancar(guion(solicita_envio=True), "hilo-invalido")
    try:
        await grafo.ainvoke(Command(resume={"aprobado": "tal vez"}), cfg)
    except ValidationError:
        print("OK  respuesta humana invalida: la rechaza el contrato de Pydantic")
        return
    raise AssertionError("Se esperaba un ValidationError")


async def prueba_serializador_redis() -> None:
    """Regresion: con Redis el estado se guarda como JSON, no como msgpack.

    Se usa el checkpointer en memoria pero con el MISMO serializador que el de
    Redis: asi se reproduce el camino real (guardar en JSON y reconstruir al
    reanudar) sin necesitar un Redis 8 levantado.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from state import Aporte, RegistroDecision

    saver = InMemorySaver(serde=serializador_redis())
    grafo, cfg, resultado = await arrancar(guion(solicita_envio=True), "hilo-serde-redis", checkpointer=saver)
    assert pedido_pendiente(resultado) is not None

    reconstruido = (await grafo.aget_state(cfg)).values
    assert all(isinstance(a, Aporte) for a in reconstruido["aportes"]), type(reconstruido["aportes"][0])
    assert all(isinstance(d, RegistroDecision) for d in reconstruido["decisiones"])

    final = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "carla"}), cfg)
    assert final["envio"] is not None and final["messages"][-1].name == "sintesis"
    print("OK  serializador de Redis: el estado vuelve como modelos y el grafo reanuda")


async def main() -> None:
    await prueba_aprobado()
    await prueba_rechazado()
    await prueba_sin_pedido_de_envio()
    await prueba_tarea_no_validada()
    await prueba_respuesta_invalida()
    await prueba_serializador_redis()
    print("\nTodas las pruebas del HITL pasaron.")


if __name__ == "__main__":
    asyncio.run(main())
