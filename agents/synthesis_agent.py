"""Nodo de sintesis final.

Se ejecuta una sola vez, cuando el Supervisor decide FINALIZAR. Redacta la
respuesta para la persona usando UNICAMENTE los datos validados que quedaron
en el estado (las salidas reales de las herramientas), no la memoria del modelo.

Esta separado del Supervisor a proposito: el Supervisor coordina, no redacta.

Si la tarea no se completo (limite de rondas o chequeos fallidos), la sintesis
lo dice de forma explicita en vez de disimularlo: es preferible una respuesta
honesta e incompleta a una completa con datos inventados.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from comun.config import Configuracion
from comun.mensajes import texto_de_mensaje
from comun.reintentos import con_reintentos
from state import Aporte, EstadoOrquestador

PROMPT_SINTESIS = """Sos el redactor final de un equipo de cocina. Recibis el
pedido de una persona y los DATOS VALIDADOS que produjeron los especialistas.

- Usa solo esos datos. No agregues recetas, numeros ni ingredientes.
- Responde en español rioplatense, claro y ordenado: primero las recetas
  elegidas, despues los numeros (calorias) y despues la lista de compras.
- En la lista de compras, indica para cada ingrediente en que recetas se usa,
  y agrega el factor de escala de cada receta: la base no tiene cantidades,
  asi que ese factor es lo que la persona necesita para saber cuanto comprar.
- Si los datos indican que la tarea NO se completo, decilo al principio con
  claridad, explica que falto y aclara que el caso queda para revision humana.
- Al final, informa en una linea el ESTADO DEL ENVIO de la lista de compras,
  tal como figura en los datos (enviada, rechazada, no enviada o no pedida).
  Nunca digas que se envio si los datos no lo confirman.
"""


def _datos_validados(aportes: list[Aporte]) -> dict[str, Any]:
    """Extrae de la evidencia los ultimos resultados exitosos de cada herramienta."""
    datos: dict[str, Any] = {}
    for aporte in aportes:
        for ev in aporte.evidencia:
            if ev.error is None and ev.herramienta in (
                "entregar_seleccion", "verificar_etiquetas",
                "calcular_nutricion_menu", "consolidar_lista_compras",
            ):
                datos[ev.herramienta] = ev.resultado  # el ultimo pisa al anterior
    return datos


def _estado_envio(estado: EstadoOrquestador) -> str:
    """PE7: resume en una linea que paso con el envio de la lista de compras."""
    if not estado.get("solicita_envio"):
        return "No se pidio enviar la lista."
    if not estado.get("task_completed"):
        return "Se pidio enviar la lista, pero NO se envio: la tarea no paso la validacion."
    envio = estado.get("envio")
    if envio is not None:
        return f"Lista ENVIADA a {envio.destino} (id {envio.envio_id})."
    decision = estado.get("decision_envio")
    if decision is not None and not decision.aprobado:
        motivo = f" Motivo: {decision.comentario}" if decision.comentario else ""
        return f"Envio RECHAZADO por {decision.revisor}; la lista no se envio.{motivo}"
    return "La lista no se envio."


def crear_nodo_sintesis(modelo: BaseChatModel, config: Configuracion):
    async def nodo_sintesis(estado: EstadoOrquestador) -> dict[str, Any]:
        pedido = next(
            (texto_de_mensaje(m) for m in estado["messages"] if isinstance(m, HumanMessage)), ""
        )
        completa = estado.get("task_completed", False)
        contexto = (
            f"PEDIDO:\n{pedido}\n\n"
            f"TAREA COMPLETADA: {'si' if completa else 'NO'}\n"
            f"MOTIVO DE CIERRE: {estado.get('motivo_cierre', '')}\n"
            f"ESTADO DEL ENVIO: {_estado_envio(estado)}\n\n"
            "DATOS VALIDADOS:\n"
            + json.dumps(_datos_validados(estado.get("aportes", [])), ensure_ascii=False, indent=1)
        )
        print("  [sintesis] redactando respuesta final...", flush=True)

        async def redactar():
            return await asyncio.wait_for(
                modelo.ainvoke([SystemMessage(PROMPT_SINTESIS), HumanMessage(contexto)]),
                timeout=config.timeout_modelo,
            )

        respuesta = await con_reintentos(redactar)
        return {"messages": [AIMessage(content=texto_de_mensaje(respuesta), name="sintesis")]}

    return nodo_sintesis
