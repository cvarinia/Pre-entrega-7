"""Esquema del estado compartido del orquestador multi-agente.

Tres ideas del Modulo 6 se materializan aca:

1. Contratos validados. Lo que un agente aporta al estado no es texto libre:
   es un `Aporte` de Pydantic. Si algo no cumple el esquema, falla en el
   momento, no tres nodos despues.

2. Aislamiento de escritura. Cada nodo escribe solo su parcela:
     - el Supervisor escribe `next_agent`, `instruccion_actual`, `ronda`,
       `decisiones`, `task_completed` y `motivo_cierre`;
     - los especialistas solo agregan un elemento a `aportes`;
     - la sintesis solo agrega el mensaje final.
   Ningun agente pisa lo que calculo otro.

3. Proveniencia (provenance). `aportes` es una lista ACUMULATIVA (reducer
   operator.add): nunca se borra nada. Cada aporte registra que agente lo
   hizo, en que ronda, con que instruccion y con que evidencia (las salidas
   reales de sus herramientas). Asi se puede rastrear de donde salio cada dato
   y detectar conflictos entre agentes.

Novedades de la PE7 (human-in-the-loop):
    - `DecisionSupervisor.solicita_envio`: el Supervisor detecta si la persona
      pidio que se envie la lista de compras (accion con efecto secundario).
    - `DecisionHumana`: el contrato de la respuesta de la persona que aprueba o
      rechaza el envio. Es lo que llega por POST /tasks/{id}/approve.
    - `ComprobanteEnvio`: la prueba de que el envio (simulado) se ejecuto.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# Nombres de los especialistas. Coinciden con los nombres de los nodos del
# grafo, asi la decision del Supervisor se mapea directo a una arista.
Especialista = Literal["investigador", "analista"]

# Opciones de ruteo del Supervisor.
Destino = Literal["investigador", "analista", "FINALIZAR"]


class Evidencia(BaseModel):
    """Una llamada a herramienta hecha por un especialista, con su resultado real."""

    herramienta: str
    argumentos: dict[str, Any] = Field(default_factory=dict)
    resultado: Any = None
    error: str | None = None


class Aporte(BaseModel):
    """Contribucion de un especialista al estado compartido."""

    agente: Especialista
    ronda: int = Field(ge=1, description="Ronda del Supervisor en la que se produjo")
    instruccion: str = Field(description="Instruccion que recibio del Supervisor")
    resumen: str = Field(description="Texto final del especialista")
    # Investigador: recetas que selecciono. Analista: recetas que analizo.
    receta_ids: list[str] = Field(default_factory=list)
    # Salidas reales de las herramientas: la prueba de donde sale cada dato.
    evidencia: list[Evidencia] = Field(default_factory=list)


class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor (y registro de cada decision)."""

    evaluacion: str = Field(
        description="Una o dos frases: que se cumple y que falta segun la rubrica."
    )
    siguiente: Destino = Field(
        description="Quien interviene ahora, o FINALIZAR si la tarea esta completa."
    )
    instruccion: str = Field(
        default="",
        description=(
            "Instruccion concreta y autocontenida para el especialista elegido. "
            "Si es un refinamiento, decir exactamente que corregir. Vacia si FINALIZAR."
        ),
    )
    # PE7: solo se tiene en cuenta cuando siguiente == "FINALIZAR".
    solicita_envio: bool = Field(
        default=False,
        description=(
            "True SOLO si la persona pidio explicitamente que se ENVIE o MANDE la "
            "lista de compras (por ejemplo al almacen o al supermercado). Pedir la "
            "lista para verla no cuenta como pedir el envio."
        ),
    )


class RegistroDecision(DecisionSupervisor):
    """Decision del Supervisor mas metadatos, para la traza."""

    ronda: int
    forzada: bool = Field(
        default=False,
        description="True si la tomo el codigo (limite de rondas) y no el LLM.",
    )


class DecisionHumana(BaseModel):
    """Respuesta de la persona que revisa una accion critica (PE7).

    Es el "input externo" que reanuda el grafo pausado. Se valida con Pydantic
    dos veces: en la API (al recibir el POST) y en el nodo de aprobacion (al
    reanudar). Si llega algo que no cumple el contrato, falla en el momento.
    """

    aprobado: bool = Field(description="True si se autoriza el envio.")
    revisor: str = Field(default="anonimo", min_length=1, max_length=80)
    comentario: str = Field(default="", max_length=500)


class ComprobanteEnvio(BaseModel):
    """Registro del envio (simulado) de la lista de compras (PE7)."""

    envio_id: str
    destino: str
    cantidad_de_items: int
    archivo: str = Field(description="Donde quedo registrado el envio simulado.")
    enviado_en: str = Field(description="Fecha y hora ISO 8601.")


class EstadoOrquestador(MessagesState):
    """Estado global del grafo.

    Hereda de `MessagesState`, que ya define:
        messages: Annotated[list[AnyMessage], add_messages]

    `messages` guarda solo la conversacion "visible": el pedido del usuario,
    una linea por cada intervencion y la respuesta final. El detalle interno
    de cada especialista (sus llamadas a herramientas) NO se vuelca aca, para
    no contaminar el contexto del resto de los agentes.
    """

    # --- Parcela del Supervisor -------------------------------------------
    next_agent: Destino
    instruccion_actual: str
    ronda: int
    decisiones: Annotated[list[RegistroDecision], operator.add]
    task_completed: bool
    motivo_cierre: str

    # --- Parcela de los especialistas (solo agregan, nunca pisan) ---------
    aportes: Annotated[list[Aporte], operator.add]

    # --- PE7: human-in-the-loop -------------------------------------------
    solicita_envio: bool                      # lo escribe el Supervisor al FINALIZAR
    decision_envio: DecisionHumana | None     # lo escribe el nodo de aprobacion
    envio: ComprobanteEnvio | None            # lo escribe el nodo de envio


# Tipos propios que viajan dentro del estado y que el checkpointer tiene que
# poder reconstruir (PE7). LangGraph solo deserializa tipos que se declaran
# explicitamente: es una medida de seguridad para no reconstruir cualquier
# clase que aparezca en la base de datos. Ver construir_grafo() en graph.py.
TIPOS_DEL_ESTADO: list[tuple[str, str]] = [
    (clase.__module__, clase.__name__)
    for clase in (Evidencia, Aporte, RegistroDecision, DecisionHumana, ComprobanteEnvio)
]
