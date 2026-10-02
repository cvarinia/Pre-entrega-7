"""Worker: saca jobs de la cola de Redis y ejecuta el grafo (PE7).

Es un proceso SEPARADO de la API. La API solo recibe pedidos y responde en
milisegundos; el trabajo pesado (30-120 s por pedido) ocurre aca.

Concurrencia: el worker no procesa los jobs de a uno. Cada job se lanza como
una tarea asyncio independiente, con un semaforo que limita cuantas corren a
la vez (MAX_TRABAJOS_CONCURRENTES). El semaforo se toma ANTES de sacar un
mensaje de la cola: si el worker esta lleno, los jobs esperan en Redis (donde
no se pierden) y no en la memoria del proceso.

Manejo de errores: cualquier excepcion dentro de un job termina en FAILED con
el motivo guardado. Sin esto, el cliente quedaria consultando para siempre.

Uso:  python -m app.worker
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from app.hitl import pedido_pendiente
from app.jobs import MensajeCola, RepositorioJobs
from app.observability import cerrar_observabilidad, contexto_de_job, iniciar_observabilidad
from comun.config import Configuracion
from comun.mensajes import texto_de_mensaje
from graph import checkpointer_redis, config_de_ejecucion, construir_grafo, estado_inicial


def resumir_resultado(final: dict[str, Any]) -> dict[str, Any]:
    """Lo que se guarda en el job al terminar: lo util para quien consulta."""
    envio = final.get("envio")
    decision = final.get("decision_envio")
    return {
        "respuesta_final": texto_de_mensaje(final["messages"][-1]),
        "recorrido": [d.siguiente for d in final.get("decisiones", [])],
        "task_completed": final.get("task_completed"),
        "motivo_cierre": final.get("motivo_cierre"),
        "solicita_envio": final.get("solicita_envio"),
        "decision_envio": decision.model_dump() if decision else None,
        "envio": envio.model_dump() if envio else None,
    }


async def procesar(
    mensaje: MensajeCola,
    grafo: CompiledStateGraph,
    jobs: RepositorioJobs,
    config: Configuracion,
) -> None:
    """Ejecuta (o reanuda) UN job. Nunca deja escapar una excepcion."""
    job_id = mensaje.job_id
    etiqueta = f"[job {job_id[:8]}]"
    inicio = time.perf_counter()
    try:
        job = await jobs.obtener(job_id)
        if job is None:
            print(f"{etiqueta} no existe (¿expiro?); se descarta el mensaje", flush=True)
            return

        await jobs.actualizar(job_id, estado="RUNNING")
        print(f"{etiqueta} RUNNING ({mensaje.accion})", flush=True)

        if mensaje.accion == "iniciar":
            entrada: Any = estado_inicial(job.pedido)
        else:
            # Reanudacion: mismo thread_id, y la decision humana como valor
            # de retorno del interrupt() que dejo el grafo en pausa.
            entrada = Command(resume=mensaje.decision.model_dump())

        # El job_id ES el thread_id: asi el checkpointer sabe que hilo retomar.
        # contexto_de_job: los spans de esta ejecucion llevan job_id, accion y etiqueta.
        with contexto_de_job(job_id, mensaje.accion, config, job.etiqueta):
            resultado = await grafo.ainvoke(entrada, config_de_ejecucion(job_id, config))
        duracion = job.duracion_segundos + (time.perf_counter() - inicio)

        pedido_aprobacion = pedido_pendiente(resultado)
        if pedido_aprobacion is not None:
            await jobs.actualizar(
                job_id, estado="AWAITING_APPROVAL",
                pedido_aprobacion=pedido_aprobacion, duracion_segundos=round(duracion, 2),
            )
            print(f"{etiqueta} AWAITING_APPROVAL (el worker queda libre)", flush=True)
        else:
            await jobs.actualizar(
                job_id, estado="DONE",
                resultado=resumir_resultado(resultado), duracion_segundos=round(duracion, 2),
            )
            print(f"{etiqueta} DONE en {duracion:.1f} s", flush=True)

    except Exception as error:  # noqa: BLE001 - a proposito: ningun error queda sin registrar
        traceback.print_exc()
        try:
            await jobs.actualizar(job_id, estado="FAILED", error=f"{type(error).__name__}: {error}")
            print(f"{etiqueta} FAILED: {type(error).__name__}", flush=True)
        except Exception:  # noqa: BLE001 - si ni Redis responde, al menos queda en consola
            traceback.print_exc()


async def bucle_worker(
    grafo: CompiledStateGraph,
    jobs: RepositorioJobs,
    config: Configuracion,
    detener: asyncio.Event | None = None,
) -> None:
    """Saca mensajes de la cola y lanza cada job como una tarea independiente."""
    detener = detener or asyncio.Event()
    semaforo = asyncio.Semaphore(config.max_trabajos_concurrentes)
    en_curso: set[asyncio.Task] = set()

    while not detener.is_set():
        await semaforo.acquire()  # primero lugar libre, despues mensaje
        mensaje = await jobs.siguiente_mensaje(espera_segundos=1)
        if mensaje is None:
            semaforo.release()
            continue

        tarea = asyncio.create_task(procesar(mensaje, grafo, jobs, config))
        en_curso.add(tarea)  # referencia fuerte: si no, la tarea podria perderse

        def al_terminar(t: asyncio.Task) -> None:
            en_curso.discard(t)
            semaforo.release()

        tarea.add_done_callback(al_terminar)

    if en_curso:
        await asyncio.gather(*en_curso, return_exceptions=True)


async def main() -> None:
    config = Configuracion.desde_entorno()
    redis = Redis.from_url(config.redis_url, decode_responses=True)
    await redis.ping()  # falla temprano y claro si Redis no esta levantado
    proveedor_trazas = iniciar_observabilidad(config)
    print(f"Worker listo | {config.proveedor}/{config.modelo} | "
          f"hasta {config.max_trabajos_concurrentes} jobs a la vez | {config.redis_url}", flush=True)
    try:
        async with checkpointer_redis(config.redis_url) as saver:
            grafo = construir_grafo(config, checkpointer=saver)
            await bucle_worker(grafo, RepositorioJobs(redis), config)
    finally:
        await redis.aclose()
        cerrar_observabilidad(proveedor_trazas)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorker detenido.")
