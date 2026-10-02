"""Construccion del modelo de lenguaje (reutilizada de la pre-entrega 5).

Este es el UNICO archivo que sabe que proveedor de LLM se esta usando. El
Supervisor, los especialistas y la sintesis reciben un `BaseChatModel` ya
armado y no saben si detras hay Gemini o Claude.

    PROVEEDOR=gemini      -> Gemini (API gratuita, usada en esta entrega)
    PROVEEDOR=anthropic   -> Claude

En esta entrega se usa un solo modelo para todos los roles por la restriccion
del free tier. La arquitectura igual permite, sin tocar el grafo, darle al
Supervisor un modelo mas capaz y a los especialistas uno mas liviano (ver
README, seccion "Especializacion de modelos").
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from comun.config import Configuracion

MAX_TOKENS_RESPUESTA = 2048


def construir_llm(config: Configuracion) -> BaseChatModel:
    """Instancia el modelo de chat del proveedor configurado.

    Las importaciones son perezosas para que el proyecto arranque aunque este
    instalado solo uno de los dos clientes.
    """
    if config.proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.modelo,
            temperature=config.temperatura,
            api_key=config.api_key,
            timeout=config.timeout_modelo,
            max_tokens=MAX_TOKENS_RESPUESTA,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.modelo,
        temperature=config.temperatura,
        google_api_key=config.api_key,
    )
