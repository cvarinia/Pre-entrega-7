"""Punto de entrada del orquestador.

Uso:
    python main.py                       # corre el pedido de demostracion
    python main.py "tu pedido aca"       # corre un pedido propio
    python main.py --diagrama            # exporta el diagrama del grafo (no usa la API)

Cada corrida deja una traza en logs/ con las decisiones del Supervisor, los
aportes de cada especialista (con su evidencia) y la respuesta final.

PE7: si el pedido incluye ENVIAR la lista de compras, el grafo se pausa y la
CLI pregunta en la terminal si se aprueba el envio. Es el mismo mecanismo que
despues usa la API (interrupt + Command(resume=...)), con un checkpointer en
memoria en lugar de Redis.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from comun.config import RAIZ_PROYECTO, Configuracion
from comun.mensajes import texto_de_mensaje
from graph import checkpointer_en_memoria, config_de_ejecucion, construir_grafo, estado_inicial
from langgraph.types import Command
from state import DecisionHumana

PEDIDO_DEMO = (
    "Armame un menu de 3 cenas vegetarianas para 6 personas. Necesito la lista "
    "de compras consolidada y saber cuantas calorias come cada persona en total."
)


def exportar_diagrama() -> None:
    """Genera docs/grafo.mmd (siempre) y docs/grafo.png (si hay internet).

    Usa el modelo simulado: el diagrama depende de la estructura del grafo, no
    del LLM, asi que no hace falta API key.
    """
    from comun.modelo_simulado import ModeloSimulado

    grafo = construir_grafo(Configuracion.para_pruebas(), modelo=ModeloSimulado())
    carpeta = RAIZ_PROYECTO / "docs"
    carpeta.mkdir(exist_ok=True)

    mermaid = grafo.get_graph().draw_mermaid()
    (carpeta / "grafo.mmd").write_text(mermaid, encoding="utf-8")
    print(f"Diagrama Mermaid guardado en {carpeta / 'grafo.mmd'}")

    try:
        # draw_mermaid_png usa el servicio mermaid.ink: necesita internet.
        (carpeta / "grafo.png").write_bytes(grafo.get_graph().draw_mermaid_png())
        print(f"Imagen guardada en {carpeta / 'grafo.png'}")
    except Exception as error:  # noqa: BLE001
        print(f"No se pudo generar el PNG ({type(error).__name__}). El .mmd sirve igual.")


def _guardar_traza(config: Configuracion, pedido: str, final: dict[str, Any], segundos: float) -> Path:
    config.ruta_logs.mkdir(exist_ok=True)
    ruta = config.ruta_logs / f"traza_{datetime.now():%Y%m%d_%H%M%S}.json"
    respuesta = next(
        (texto_de_mensaje(m) for m in reversed(final["messages"]) if getattr(m, "name", "") == "sintesis"), ""
    )
    traza = {
        "pedido": pedido,
        "proveedor": config.proveedor,
        "modelo": config.modelo,
        "duracion_segundos": round(segundos, 1),
        "task_completed": final.get("task_completed"),
        "motivo_cierre": final.get("motivo_cierre"),
        "recorrido": [d.siguiente for d in final.get("decisiones", [])],
        "decisiones_supervisor": [d.model_dump() for d in final.get("decisiones", [])],
        "aportes": [a.model_dump() for a in final.get("aportes", [])],
        "respuesta_final": respuesta,
        "solicita_envio": final.get("solicita_envio"),
        "decision_envio": final["decision_envio"].model_dump() if final.get("decision_envio") else None,
        "envio": final["envio"].model_dump() if final.get("envio") else None,
    }
    ruta.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


async def _preguntar_aprobacion(pedido_aprobacion: dict[str, Any]) -> DecisionHumana:
    """Muestra el pedido de aprobacion y lee la respuesta en la terminal."""
    print("\n" + "-" * 70)
    print(f"APROBACION REQUERIDA: {pedido_aprobacion['pregunta']}")
    print(f"Destino: {pedido_aprobacion['destino']} | {pedido_aprobacion['cantidad_de_items']} items")
    print("Items:", ", ".join(pedido_aprobacion["items"]))
    print("-" * 70)
    # input() bloquea: se corre en un hilo aparte para no frenar el event loop.
    respuesta = await asyncio.to_thread(input, "¿Aprobas el envio? (s/n): ")
    aprobado = respuesta.strip().lower() in ("s", "si", "sí", "y")
    comentario = "" if aprobado else await asyncio.to_thread(input, "Motivo (opcional): ")
    return DecisionHumana(aprobado=aprobado, revisor="cli", comentario=comentario.strip())


async def ejecutar(pedido: str) -> dict[str, Any]:
    config = Configuracion.desde_entorno()
    grafo = construir_grafo(config, checkpointer=checkpointer_en_memoria())
    cfg = config_de_ejecucion(f"cli-{uuid.uuid4().hex[:8]}", config)
    print(f"Modelo: {config.proveedor} / {config.modelo}")
    print(f"Pedido: {pedido}\n")

    inicio = time.perf_counter()
    final: dict[str, Any] = {}
    entrada: Any = estado_inicial(pedido)
    while True:
        # "updates" muestra que nodo termino; "values" trae el estado completo.
        async for modo, datos in grafo.astream(entrada, cfg, stream_mode=["updates", "values"]):
            if modo == "updates":
                for nodo in datos:
                    if nodo != "__interrupt__":
                        print(f"<- termino: {nodo}\n", flush=True)
            else:
                final = datos

        # ¿Termino o quedo pausado en el HITL?
        snapshot = await grafo.aget_state(cfg)
        if not snapshot.interrupts:
            break
        decision = await _preguntar_aprobacion(snapshot.interrupts[0].value)
        entrada = Command(resume=decision.model_dump())  # se reanuda en el mismo thread_id
    segundos = time.perf_counter() - inicio

    print("=" * 70)
    print(texto_de_mensaje(final["messages"][-1]))
    print("=" * 70)
    print(f"Recorrido: {' -> '.join(d.siguiente for d in final['decisiones'])}")
    print(f"Tarea completa: {final['task_completed']} | {final['motivo_cierre']}")
    print(f"Traza guardada en: {_guardar_traza(config, pedido, final, segundos)}")
    return final


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    if argumentos == ["--diagrama"]:
        exportar_diagrama()
    else:
        asyncio.run(ejecutar(" ".join(argumentos) or PEDIDO_DEMO))
