"""Observabilidad con Arize Phoenix (PE7).

Phoenix recibe TRAZAS: el arbol de todo lo que paso en una ejecucion. Cada
nodo del grafo, cada llamada al LLM y cada herramienta es un SPAN dentro de
ese arbol, con su duracion, sus tokens y su costo.

Como llegan las trazas a Phoenix (estandar OpenTelemetry / OpenInference):

    1. Instrumentacion: LangChainInstrumentor "envuelve" LangChain y LangGraph.
       No hace falta decorar funciones a mano: cada Runnable (nodos del grafo,
       modelos, herramientas) genera su span automaticamente.
    2. Exportador: register() configura a donde se mandan (el colector de Phoenix).
    3. Dashboard: http://localhost:6006

Ademas, cada job agrega CONTEXTO a sus spans (contexto_de_job):
    - session.id = job_id -> la ejecucion inicial y la reanudacion despues de
      la aprobacion humana quedan agrupadas en la misma sesion.
    - metadata y tags -> permiten filtrar en el dashboard (por ejemplo, solo
      las 5 ejecuciones de la prueba de carga).

Las trazas se generan en el WORKER (ahi corre el grafo), no en la API.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from collections.abc import Iterator
from typing import Any

from comun.config import Configuracion

# Version de los prompts del sistema. Si se cambia un prompt, se sube este
# numero: asi, ante un cambio de comportamiento, en el dashboard se puede ver
# que version estaba activa (buena practica de la teoria del modulo).
VERSION_PROMPTS = "pe7-v1"


def iniciar_observabilidad(config: Configuracion) -> Any | None:
    """Conecta la instrumentacion con Phoenix. Devuelve el tracer provider (o None).

    Si OBSERVABILIDAD=false en el .env, no hace nada: el sistema funciona igual,
    solo que sin mandar trazas.
    """
    if not config.observabilidad:
        print("Observabilidad desactivada (OBSERVABILIDAD=false).", flush=True)
        return None

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    proveedor_trazas = register(
        project_name=config.proyecto_phoenix,
        endpoint=config.phoenix_endpoint,
        batch=True,       # manda los spans en lotes, sin frenar la ejecucion
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=proveedor_trazas)
    print(f"Observabilidad: trazas -> {config.phoenix_endpoint} "
          f"(proyecto '{config.proyecto_phoenix}')", flush=True)
    return proveedor_trazas


def cerrar_observabilidad(proveedor_trazas: Any | None) -> None:
    """Manda los spans que quedaron en el lote antes de que termine el proceso."""
    if proveedor_trazas is not None:
        proveedor_trazas.force_flush()
        proveedor_trazas.shutdown()


@contextmanager
def contexto_de_job(
    job_id: str, accion: str, config: Configuracion, etiqueta: str | None = None
) -> Iterator[None]:
    """Todo span creado dentro de este bloque lleva el contexto del job.

    Funciona aunque haya varios jobs a la vez: cada tarea asyncio tiene su
    propia copia del contexto.
    """
    from openinference.instrumentation import using_metadata, using_session, using_tags

    metadata = {
        "job_id": job_id,
        "accion": accion,  # "iniciar" o "reanudar"
        "proveedor": config.proveedor,
        "modelo": config.modelo,
        "version_prompts": VERSION_PROMPTS,
    }
    tags = [accion, VERSION_PROMPTS]
    if etiqueta:
        metadata["etiqueta"] = etiqueta
        tags.append(etiqueta)

    with ExitStack() as pila:
        pila.enter_context(using_session(job_id))
        pila.enter_context(using_metadata(metadata))
        pila.enter_context(using_tags(tags))
        yield
