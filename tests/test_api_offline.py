"""Pruebas de la API + worker + Redis SIN consumir la API del LLM.

Necesitan Redis levantado (docker compose up -d redis). Usan la base 15 de
Redis y la vacian al empezar, para no mezclarse con los datos de desarrollo.

Lo simulado: el LLM (ModeloSimulado) y el checkpointer (en memoria). Lo real:
FastAPI, Redis como registro y cola, el worker con su semaforo y su manejo de
errores.

Escenarios:
    1. Flujo completo con aprobacion: PENDING -> AWAITING_APPROVAL -> approve -> DONE.
    2. Errores del cliente: 404, 409 (aprobar dos veces / en otro estado), 422.
    3. Falla del agente -> FAILED con el motivo (no hay polling infinito).
    4. Concurrencia: 5 jobs de 1 s terminan en ~1 s, no en 5 s.

Uso:  python -m tests.test_api_offline
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from redis.asyncio import Redis  # noqa: E402

from app.jobs import RepositorioJobs  # noqa: E402
from app.main import app  # noqa: E402
from app.worker import bucle_worker  # noqa: E402
from comun.modelo_simulado import ModeloSimulado  # noqa: E402
from graph import checkpointer_en_memoria, construir_grafo  # noqa: E402
from tests.test_hitl_offline import PEDIDO, config_de_prueba, guion  # noqa: E402

REDIS_URL_PRUEBAS = "redis://localhost:6379/15"


async def esperar_estado(cliente: httpx.AsyncClient, job_id: str, estados: set[str], limite: float = 20) -> dict:
    """Polling: consulta cada 0,2 s hasta que el job llegue a uno de `estados`."""
    fin = time.monotonic() + limite
    while time.monotonic() < fin:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] in estados:
            return job
        await asyncio.sleep(0.2)
    raise AssertionError(f"El job {job_id} no llego a {estados}; ultimo estado: {job['estado']}")


class Entorno:
    """Levanta: Redis de pruebas + cliente HTTP contra la app + un worker en segundo plano."""

    def __init__(self, grafo, config) -> None:
        self.grafo, self.config = grafo, config

    async def __aenter__(self) -> httpx.AsyncClient:
        self.redis = Redis.from_url(REDIS_URL_PRUEBAS, decode_responses=True)
        await self.redis.flushdb()
        jobs = RepositorioJobs(self.redis)
        app.state.redis, app.state.jobs = self.redis, jobs  # lo que haria el lifespan
        self.detener = asyncio.Event()
        self.worker = asyncio.create_task(bucle_worker(self.grafo, jobs, self.config, self.detener))
        self.cliente = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
        return self.cliente

    async def __aexit__(self, *exc) -> None:
        self.detener.set()
        await self.worker
        await self.cliente.aclose()
        await self.redis.aclose()


def grafo_simulado(modelo: ModeloSimulado):
    config = config_de_prueba()
    return construir_grafo(config, modelo=modelo, checkpointer=checkpointer_en_memoria()), config


async def prueba_flujo_con_aprobacion() -> None:
    grafo, config = grafo_simulado(guion(solicita_envio=True))
    async with Entorno(grafo, config) as cliente:
        t0 = time.perf_counter()
        creado = await cliente.post("/tasks", json={"pedido": PEDIDO})
        demora_ms = (time.perf_counter() - t0) * 1000
        assert creado.status_code == 202, creado.text
        job_id = creado.json()["job_id"]

        pausado = await esperar_estado(cliente, job_id, {"AWAITING_APPROVAL", "FAILED"})
        assert pausado["estado"] == "AWAITING_APPROVAL", pausado
        assert pausado["pedido_aprobacion"]["accion"] == "enviar_lista_compras"

        respuesta = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "carla"})
        assert respuesta.status_code == 202, respuesta.text

        final = await esperar_estado(cliente, job_id, {"DONE", "FAILED"})
        assert final["estado"] == "DONE", final
        assert final["resultado"]["envio"] is not None
        assert final["resultado"]["recorrido"] == ["investigador", "analista", "FINALIZAR"]
        print(f"OK  flujo completo: POST respondio en {demora_ms:.0f} ms -> AWAITING_APPROVAL -> approve -> DONE")

        # Errores del cliente, con el mismo job ya terminado:
        assert (await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409
        assert (await cliente.get("/tasks/no-existe")).status_code == 404
        assert (await cliente.post("/tasks", json={"pedido": "x"})).status_code == 422
        print("OK  errores del cliente: 409 (aprobar un job terminado), 404, 422")


async def prueba_respuesta_invalida_en_approve() -> None:
    grafo, config = grafo_simulado(guion(solicita_envio=True))
    async with Entorno(grafo, config) as cliente:
        job_id = (await cliente.post("/tasks", json={"pedido": PEDIDO})).json()["job_id"]
        await esperar_estado(cliente, job_id, {"AWAITING_APPROVAL"})
        invalida = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": "tal vez"})
        assert invalida.status_code == 422, invalida.text
        sigue = (await cliente.get(f"/tasks/{job_id}")).json()
        assert sigue["estado"] == "AWAITING_APPROVAL"  # una respuesta invalida no toca el job
        print("OK  approve con cuerpo invalido: 422 y el job sigue esperando")


async def prueba_falla_del_agente() -> None:
    # Guion vacio: el Supervisor no tiene decisiones -> excepcion dentro del grafo.
    grafo, config = grafo_simulado(ModeloSimulado(respuestas=[AIMessage(content="-")], decisiones=[]))
    async with Entorno(grafo, config) as cliente:
        job_id = (await cliente.post("/tasks", json={"pedido": PEDIDO})).json()["job_id"]
        final = await esperar_estado(cliente, job_id, {"DONE", "FAILED"})
        assert final["estado"] == "FAILED" and final["error"], final
        print(f"OK  falla del agente -> FAILED ({final['error'][:60]}...)")


class GrafoLento:
    """Grafo falso que tarda 1 s: sirve para medir la concurrencia del worker."""

    async def ainvoke(self, entrada, config):
        await asyncio.sleep(1)
        return {"messages": [AIMessage(content="listo")], "decisiones": []}


async def prueba_concurrencia() -> None:
    config = replace(config_de_prueba(), max_trabajos_concurrentes=5)
    async with Entorno(GrafoLento(), config) as cliente:
        t0 = time.perf_counter()
        creados = await asyncio.gather(*[cliente.post("/tasks", json={"pedido": f"pedido {i}"}) for i in range(5)])
        ids = [c.json()["job_id"] for c in creados]
        await asyncio.gather(*[esperar_estado(cliente, i, {"DONE"}) for i in ids])
        total = time.perf_counter() - t0
        assert total < 2.5, f"Tardo {total:.1f} s: los jobs corrieron en fila"
        print(f"OK  concurrencia: 5 jobs de 1 s terminaron en {total:.1f} s (en fila serian 5 s)")


async def main() -> None:
    await prueba_flujo_con_aprobacion()
    await prueba_respuesta_invalida_en_approve()
    await prueba_falla_del_agente()
    await prueba_concurrencia()
    print("\nTodas las pruebas de la API pasaron.")


if __name__ == "__main__":
    asyncio.run(main())
