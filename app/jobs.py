"""Estado de los jobs y cola de trabajo en Redis (PE7).

Redis cumple dos roles en este archivo (el tercero, los checkpoints del grafo,
lo maneja LangGraph con su propio checkpointer):

    1. REGISTRO: cada job es una clave  job:<id>  con su estado en JSON.
       Es lo que consulta GET /tasks/{id}.
    2. COLA: una lista  cola:jobs  con mensajes "que hay que hacer".
       La API empuja (RPUSH) y el worker saca (BLPOP) -> primero en entrar,
       primero en salir.

Ciclo de vida de un job:

    PENDING ──> RUNNING ──> DONE
                  │  └────> FAILED
                  └──> AWAITING_APPROVAL ──(approve)──> PENDING ──> RUNNING ...

Regla de la teoria que se respeta aca: el registro se escribe ANTES de
encolar. Si fuera al reves, el worker podria tomar el job antes de que exista
y el cliente recibiria un 404 al consultar.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from redis.asyncio import Redis

from state import DecisionHumana

EstadoJob = Literal["PENDING", "RUNNING", "AWAITING_APPROVAL", "DONE", "FAILED"]

COLA = "cola:jobs"
TTL_JOB_SEGUNDOS = 7 * 24 * 3600  # los registros se borran solos a la semana


def _ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


class Job(BaseModel):
    """Lo que la API devuelve al consultar un job."""

    job_id: str
    estado: EstadoJob
    pedido: str
    # Etiqueta libre para agrupar jobs (por ejemplo, la prueba de carga).
    # Viaja a las trazas como metadata y tag: permite filtrarlas en Phoenix.
    etiqueta: str | None = None
    creado_en: str
    actualizado_en: str
    # Solo con AWAITING_APPROVAL: lo que tiene que revisar la persona.
    pedido_aprobacion: dict[str, Any] | None = None
    # Solo con DONE: respuesta final y resumen de la ejecucion.
    resultado: dict[str, Any] | None = None
    # Solo con FAILED: que salio mal.
    error: str | None = None
    # Tiempo de ejecucion del grafo, SIN contar la espera de la aprobacion
    # humana. Sirve para calcular la latencia p95 por fuera del dashboard.
    duracion_segundos: float = 0.0


class MensajeCola(BaseModel):
    """Lo que viaja por la cola: que job y que hacer con el."""

    job_id: str
    accion: Literal["iniciar", "reanudar"]
    decision: DecisionHumana | None = None  # solo para "reanudar"


class RepositorioJobs:
    """Toda la interaccion con Redis para jobs pasa por aca."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    # --- Registro -----------------------------------------------------------

    async def obtener(self, job_id: str) -> Job | None:
        crudo = await self.redis.get(_clave(job_id))
        return Job.model_validate_json(crudo) if crudo else None

    async def _guardar(self, job: Job) -> None:
        job.actualizado_en = _ahora()
        await self.redis.set(_clave(job.job_id), job.model_dump_json(), ex=TTL_JOB_SEGUNDOS)

    async def actualizar(self, job_id: str, **cambios: Any) -> Job:
        job = await self.obtener(job_id)
        if job is None:
            raise KeyError(f"No existe el job {job_id}")
        job = job.model_copy(update=cambios)
        await self._guardar(job)
        return job

    # --- Operaciones que usa la API ------------------------------------------

    async def crear(self, pedido: str, etiqueta: str | None = None) -> Job:
        """Registra el job como PENDING y DESPUES lo encola (en ese orden)."""
        job = Job(
            job_id=uuid.uuid4().hex,
            estado="PENDING",
            pedido=pedido,
            etiqueta=etiqueta,
            creado_en=_ahora(),
            actualizado_en=_ahora(),
        )
        await self._guardar(job)                                           # 1) registro
        await self._encolar(MensajeCola(job_id=job.job_id, accion="iniciar"))  # 2) cola
        return job

    async def responder_aprobacion(self, job_id: str, decision: DecisionHumana) -> bool:
        """Guarda la decision humana y encola la reanudacion.

        Devuelve False si el job ya habia sido respondido. El candado
        (SET ... NX: "escribir solo si no existe") evita que dos aprobaciones
        simultaneas reanuden el mismo grafo dos veces.
        """
        primera_vez = await self.redis.set(
            f"{_clave(job_id)}:decision", decision.model_dump_json(), nx=True, ex=TTL_JOB_SEGUNDOS
        )
        if not primera_vez:
            return False
        await self.actualizar(job_id, estado="PENDING", pedido_aprobacion=None)
        await self._encolar(MensajeCola(job_id=job_id, accion="reanudar", decision=decision))
        return True

    # --- Cola ---------------------------------------------------------------

    async def _encolar(self, mensaje: MensajeCola) -> None:
        await self.redis.rpush(COLA, mensaje.model_dump_json())

    async def siguiente_mensaje(self, espera_segundos: int = 5) -> MensajeCola | None:
        """Espera hasta `espera_segundos` a que haya algo en la cola (lo usa el worker).

        BLPOP bloquea la CONEXION con Redis, no el event loop: mientras espera,
        los demas jobs del worker siguen corriendo.
        """
        item = await self.redis.blpop([COLA], timeout=espera_segundos)
        if item is None:
            return None
        _, crudo = item
        return MensajeCola.model_validate(json.loads(crudo))
