"""Utilidades para leer el contenido de un mensaje del modelo.

Los modelos modernos no siempre devuelven texto plano en `mensaje.content`:
Gemini 3, por ejemplo, devuelve una lista de bloques
`[{"type": "text", "text": "..."}, {...firmas internas...}]`.

Si uno hace `str(contenido)` sobre eso, termina guardando en la traza la
representacion de Python entera, con comillas simples y bloques de firma
ilegibles. Esta funcion extrae solo el texto, y funciona igual si el modelo
devuelve un string comun.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage


def texto_de_contenido(contenido: Any) -> str:
    """Devuelve el texto legible de un `content`, sea string o lista de bloques."""
    if isinstance(contenido, str):
        return contenido

    if isinstance(contenido, list):
        partes: list[str] = []
        for bloque in contenido:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict):
                texto = bloque.get("text")
                if isinstance(texto, str) and texto:
                    partes.append(texto)
        return "\n".join(partes).strip()

    return str(contenido)


def texto_de_mensaje(mensaje: BaseMessage) -> str:
    """Atajo: extrae el texto legible de un mensaje de LangChain."""
    return texto_de_contenido(mensaje.content)
