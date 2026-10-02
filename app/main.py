"""API REST del orquestador (PE7).

La API NUNCA ejecuta el grafo: registra el pedido, lo encola en Redis y
devuelve un job_id en milisegundos. El trabajo lo hace el worker
(python -m app.worker), que es otro proceso.

Endpoints:
    POST /tasks                 -> crea el job (202 + job_id)
    GET  /tasks/{job_id}        -> estado del job (el cliente consulta cada tanto: polling)
    POST /tasks/{job_id}/approve -> respuesta humana para un job en AWAITING_APPROVAL
    GET  /health                -> ¿la API y Redis responden?

Todos los endpoints son async y solo hablan con Redis de forma asincrona:
el event loop nunca queda bloqueado.

Uso:  uvicorn app.main:app --reload
Documentacion interactiva: http://localhost:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.jobs import Job, RepositorioJobs
from comun.config import Configuracion
from state import DecisionHumana


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    """Abre la conexion a Redis al arrancar y la cierra al apagar."""
    config = Configuracion.desde_entorno()
    redis = Redis.from_url(config.redis_url, decode_responses=True)
    await redis.ping()  # si Redis no esta levantado, la API no arranca (falla clara)
    app.state.redis = redis
    app.state.jobs = RepositorioJobs(redis)
    yield
    await redis.aclose()


app = FastAPI(
    title="Orquestador de menus - API",
    description="Pre-entrega 7: sistema multi-agente con cola en Redis y aprobacion humana.",
    version="0.7.0",
    lifespan=ciclo_de_vida,
)


# --------------------------------------------------------------------------
# Contratos de entrada y salida
# --------------------------------------------------------------------------


class PedidoTarea(BaseModel):
    pedido: str = Field(
        min_length=5, max_length=1000,
        examples=["Armame 3 cenas vegetarianas para 6 personas y mandale la lista de compras al almacen"],
    )
    etiqueta: str | None = Field(
        default=None, max_length=60, pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Opcional. Agrupa jobs en el dashboard de trazas (por ejemplo, 'carga-1').",
    )


class TareaCreada(BaseModel):
    job_id: str
    estado: str
    consultar_en: str


def _jobs(request: Request) -> RepositorioJobs:
    return request.app.state.jobs


async def _job_o_404(request: Request, job_id: str) -> Job:
    job = await _jobs(request).obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    return job


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TareaCreada)
async def crear_tarea(cuerpo: PedidoTarea, request: Request) -> TareaCreada:
    """Encola el pedido y devuelve el job_id al instante (202 = aceptado, en proceso)."""
    job = await _jobs(request).crear(cuerpo.pedido, cuerpo.etiqueta)
    return TareaCreada(job_id=job.job_id, estado=job.estado, consultar_en=f"/tasks/{job.job_id}")


@app.get("/tasks/{job_id}", response_model=Job)
async def consultar_tarea(job_id: str, request: Request) -> Job:
    """Estado actual del job. Si esta en AWAITING_APPROVAL, incluye que hay que aprobar."""
    return await _job_o_404(request, job_id)


@app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=Job)
async def aprobar_tarea(job_id: str, decision: DecisionHumana, request: Request) -> Job:
    """Respuesta humana (aprobar o rechazar) para un job pausado.

    El cuerpo se valida con el mismo contrato (DecisionHumana) que usa el nodo
    de aprobacion del grafo: lo que no cumple, se rechaza aca con un 422.
    """
    job = await _job_o_404(request, job_id)
    if job.estado != "AWAITING_APPROVAL":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job esta en {job.estado}: solo se puede aprobar en AWAITING_APPROVAL.",
        )
    if not await _jobs(request).responder_aprobacion(job_id, decision):
        raise HTTPException(status.HTTP_409_CONFLICT, "Este job ya fue respondido.")
    return await _job_o_404(request, job_id)


@app.get("/health")
async def salud(request: Request) -> dict[str, str]:
    await request.app.state.redis.ping()
    return {"api": "ok", "redis": "ok"}
