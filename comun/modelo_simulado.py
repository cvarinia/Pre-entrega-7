"""Modelo de chat simulado: reemplaza al LLM con respuestas guionadas.

Sirve para dos cosas sin gastar una sola llamada a la API:
    - los tests offline (verifican el cableado del grafo);
    - generar el diagrama del grafo sin tener una API key configurada.

Imita lo unico que el sistema le pide al modelo:
    - `ainvoke` / `invoke` -> devuelve el siguiente AIMessage del guion
      (usado por los especialistas y la sintesis);
    - `bind_tools` -> se devuelve a si mismo (lo llama create_react_agent);
    - `with_structured_output` -> devuelve la siguiente decision del guion
      (usado por el Supervisor).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


class ModeloSimulado(BaseChatModel):
    respuestas: list[AIMessage] = []
    decisiones: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "modelo-simulado"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        mensaje = self.respuestas.pop(0) if self.respuestas else AIMessage(content="(sin mas respuestas simuladas)")
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001
        def siguiente_decision(_: Any) -> Any:
            if not self.decisiones:
                raise RuntimeError("El guion simulado se quedo sin decisiones del Supervisor.")
            return self.decisiones.pop(0)

        return RunnableLambda(siguiente_decision)
