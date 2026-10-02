"""Piezas comunes a los dos especialistas.

Cada especialista es un agente ReAct propio (creado con `create_react_agent`)
que corre ADENTRO de un nodo del grafo principal. Ese sub-agente arranca
siempre con un historial limpio: solo recibe la instruccion del Supervisor.
Es la defensa contra la "contaminacion de contexto" que advierte la consigna.

Cuando el sub-agente termina, de su historial interno se extraen dos cosas:
    - el texto final (resumen), y
    - la EVIDENCIA: cada herramienta que llamo, con sus argumentos y su
      resultado real. Eso es lo que se guarda como proveniencia.
El resto del historial interno se descarta: no viaja al estado global.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from comun.config import Configuracion
from comun.mensajes import texto_de_mensaje
from comun.reintentos import con_reintentos
from state import Evidencia


def _parsear(contenido: Any) -> Any:
    """Las tools devuelven dicts; en el ToolMessage llegan como JSON en texto."""
    if isinstance(contenido, str):
        try:
            return json.loads(contenido)
        except json.JSONDecodeError:
            return contenido
    return contenido


def extraer_evidencia(mensajes: list[AnyMessage]) -> list[Evidencia]:
    """Arma la lista de evidencia cruzando cada pedido de tool con su respuesta."""
    argumentos_por_id: dict[str, dict[str, Any]] = {}
    for mensaje in mensajes:
        if isinstance(mensaje, AIMessage):
            for llamada in mensaje.tool_calls:
                argumentos_por_id[llamada["id"]] = llamada["args"]

    evidencia: list[Evidencia] = []
    for mensaje in mensajes:
        if isinstance(mensaje, ToolMessage):
            resultado = _parsear(mensaje.content)
            error = None
            if isinstance(resultado, dict) and "error" in resultado:
                error = str(resultado["error"])
            elif mensaje.status == "error":
                error = texto_de_mensaje(mensaje)
            evidencia.append(
                Evidencia(
                    herramienta=mensaje.name or "desconocida",
                    argumentos=argumentos_por_id.get(mensaje.tool_call_id, {}),
                    resultado=resultado,
                    error=error,
                )
            )
    return evidencia


def texto_final(mensajes: list[AnyMessage]) -> str:
    """Texto del ultimo mensaje del modelo que no sea un pedido de herramientas."""
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, AIMessage) and not mensaje.tool_calls:
            texto = texto_de_mensaje(mensaje)
            if texto:
                return texto
    return "(el especialista no dejo un resumen en texto)"


async def ejecutar_especialista(
    agente: CompiledStateGraph, instruccion: str, config: Configuracion
) -> list[AnyMessage]:
    """Corre el sub-agente ReAct con un historial limpio y devuelve sus mensajes.

    - Historial limpio: el unico mensaje de entrada es la instruccion.
    - `recursion_limit`: techo de pasos internos del especialista.
    - `con_reintentos`: tolera errores temporales del proveedor (429, 503).
    - `wait_for`: evita que una llamada colgada deje el nodo esperando para siempre.
    """

    async def correr() -> dict[str, Any]:
        return await asyncio.wait_for(
            agente.ainvoke(
                {"messages": [HumanMessage(content=instruccion)]},
                {"recursion_limit": config.limite_pasos_especialista},
            ),
            timeout=config.timeout_modelo * 3,
        )

    resultado = await con_reintentos(correr)
    return list(resultado["messages"])
