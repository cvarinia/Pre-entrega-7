"""Human-in-the-loop: aprobacion humana antes de una accion critica (PE7).

La accion critica de este sistema es ENVIAR la lista de compras (simulado: al
"almacen"). Es un efecto secundario: una vez enviada, no se puede deshacer. Por
eso el grafo se detiene y espera que una persona la apruebe.

Se usan DOS nodos separados a proposito:

    aprobacion_envio   -> solo pregunta (interrupt). No tiene efectos secundarios.
    enviar_lista       -> solo ejecuta el envio. Se llega unicamente si se aprobo.

Por que separarlos: cuando el grafo se reanuda, LangGraph vuelve a ejecutar
DESDE EL PRINCIPIO el nodo que llamo a interrupt(). Si el envio estuviera en el
mismo nodo, antes de la pausa, se ejecutaria dos veces. Regla: en un nodo con
interrupt(), nada con efectos secundarios antes de la pausa.

Como funciona la pausa:
    1. `interrupt(pedido)` guarda el estado en el checkpointer y corta la
       ejecucion. `ainvoke` devuelve el estado con la clave "__interrupt__".
    2. Nadie se queda esperando: el worker queda libre para otros jobs.
    3. Cuando llega la respuesta (POST /tasks/{id}/approve), se reanuda con
       `Command(resume=respuesta)` y el MISMO thread_id. interrupt() devuelve
       esa respuesta y el nodo sigue.
Sin checkpointer esto es imposible: no habria donde guardar el estado pausado.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from comun.config import Configuracion
from state import Aporte, ComprobanteEnvio, DecisionHumana, EstadoOrquestador

ACCION_CRITICA = "enviar_lista_compras"
DESTINO_SIMULADO = "Almacen del barrio (simulado)"


def lista_validada(aportes: list[Aporte]) -> dict[str, Any] | None:
    """Ultima lista de compras calculada SIN error por el analista (proveniencia)."""
    lista = None
    for aporte in aportes:
        for ev in aporte.evidencia:
            if ev.herramienta == "consolidar_lista_compras" and ev.error is None:
                lista = ev.resultado  # la ultima pisa a la anterior
    return lista


# --------------------------------------------------------------------------
# Nodo 1: la pausa
# --------------------------------------------------------------------------


def crear_nodo_aprobacion():
    """Devuelve el nodo que pausa el grafo hasta recibir una DecisionHumana."""

    async def nodo_aprobacion(estado: EstadoOrquestador) -> dict[str, Any]:
        lista = lista_validada(estado.get("aportes", [])) or {}

        # Lo que ve la persona que tiene que aprobar. Tiene que ser serializable
        # a JSON: viaja al checkpointer y despues a la API.
        pedido_aprobacion = {
            "accion": ACCION_CRITICA,
            "destino": DESTINO_SIMULADO,
            "pregunta": "¿Autorizas el envio de esta lista de compras?",
            "cantidad_de_items": lista.get("cantidad_de_items", 0),
            "items": [item["ingrediente"] for item in lista.get("items", [])],
        }
        print(f"  [aprobacion] pausa: esperando decision humana ({ACCION_CRITICA})", flush=True)

        # --- Aca se detiene el grafo. Todo lo de arriba se vuelve a ejecutar
        # al reanudar, por eso no hay efectos secundarios antes de esta linea.
        respuesta = interrupt(pedido_aprobacion)

        # Al reanudar: validacion del contrato. Si la respuesta no cumple,
        # ValidationError -> el job termina en FAILED con el motivo.
        decision = DecisionHumana.model_validate(respuesta)
        veredicto = "APROBADO" if decision.aprobado else "RECHAZADO"
        print(f"  [aprobacion] {veredicto} por {decision.revisor}", flush=True)

        return {
            "decision_envio": decision,
            "messages": [AIMessage(
                content=f"Envio {veredicto.lower()} por {decision.revisor}. {decision.comentario}".strip(),
                name="aprobacion",
            )],
        }

    return nodo_aprobacion


def rutear_tras_aprobacion(estado: EstadoOrquestador) -> str:
    """Arista condicional: solo se envia si hubo aprobacion explicita."""
    decision = estado.get("decision_envio")
    return "enviar_lista" if decision is not None and decision.aprobado else "sintesis"


# --------------------------------------------------------------------------
# Nodo 2: el efecto secundario
# --------------------------------------------------------------------------


def _escribir_json(ruta: Path, datos: dict[str, Any]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def crear_nodo_envio(config: Configuracion):
    """Devuelve el nodo que ejecuta el envio simulado.

    El "envio" se registra como un archivo JSON en logs/envios/. En un sistema
    real seria un mail, un pedido a una API de un supermercado, etc.
    """

    async def nodo_envio(estado: EstadoOrquestador) -> dict[str, Any]:
        lista = lista_validada(estado.get("aportes", [])) or {}
        envio_id = uuid.uuid4().hex[:8]
        ruta = config.ruta_logs / "envios" / f"envio_{envio_id}.json"

        comprobante = ComprobanteEnvio(
            envio_id=envio_id,
            destino=DESTINO_SIMULADO,
            cantidad_de_items=lista.get("cantidad_de_items", 0),
            archivo=str(ruta.relative_to(config.ruta_logs.parent)),
            enviado_en=datetime.now().isoformat(timespec="seconds"),
        )
        decision = estado.get("decision_envio")

        # Escribir un archivo es I/O bloqueante: se manda a un hilo aparte para
        # no frenar el event loop (el mismo error que advierte la consigna).
        await asyncio.to_thread(_escribir_json, ruta, {
            **comprobante.model_dump(),
            "aprobado_por": decision.revisor if decision else None,
            "lista": lista,
        })
        print(f"  [envio] lista enviada a {DESTINO_SIMULADO} -> {comprobante.archivo}", flush=True)

        return {
            "envio": comprobante,
            "messages": [AIMessage(
                content=f"Lista de compras enviada ({comprobante.cantidad_de_items} items), id {envio_id}.",
                name="envio",
            )],
        }

    return nodo_envio


# --------------------------------------------------------------------------
# Ayuda para quien ejecuta el grafo (CLI hoy, worker despues)
# --------------------------------------------------------------------------


def pedido_pendiente(resultado: dict[str, Any]) -> dict[str, Any] | None:
    """Si el grafo quedo pausado, devuelve lo que se le pregunta a la persona.

    `ainvoke` devuelve el estado; si hubo una pausa, trae ademas la clave
    "__interrupt__" con los objetos Interrupt (su `.value` es el pedido).
    """
    interrupciones = resultado.get("__interrupt__") or []
    return interrupciones[0].value if interrupciones else None
