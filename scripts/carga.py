"""Prueba de carga: 5 peticiones concurrentes contra la API (PE7).

Que hace:
    1. Manda 5 pedidos DISTINTOS a POST /tasks al mismo tiempo, todos con la
       misma etiqueta (ej. "carga-20261002-153000").
    2. Hace polling de los 5 jobs hasta que terminan.
    3. Le pide a Phoenix las metricas de ESAS 5 trazas (filtradas por la
       etiqueta): costo total, costo por ejecucion y latencia p50 / p95.
    4. Guarda todo en logs/carga_<etiqueta>.json.

Ninguno de los pedidos pide ENVIAR la lista: asi no se dispara la aprobacion
humana y la latencia mide al sistema, no el tiempo que tarda una persona.

Requisitos: Redis, Phoenix, el worker y la API levantados.

Uso:
    python scripts/carga.py
    python scripts/carga.py --api http://localhost:8000 --phoenix http://localhost:6006
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent.parent

PEDIDOS = [
    "Armame 3 cenas vegetarianas para 6 personas, con las calorias y la lista de compras.",
    "Necesito 2 platos sin gluten para 4 personas, con calorias por porcion y lista de compras.",
    "Proponeme 3 recetas sin lacteos para 8 personas y calcula las calorias totales.",
    "Quiero 2 recetas veganas para 4 personas, con la lista de compras.",
    "Buscame 3 recetas sin gluten y sin lacteos para 5 personas y arma la lista de compras.",
]
ESTADOS_FINALES = {"DONE", "FAILED", "AWAITING_APPROVAL"}


# --------------------------------------------------------------------------
# 1 y 2: lanzar y esperar
# --------------------------------------------------------------------------


async def lanzar_y_esperar(cliente: httpx.AsyncClient, pedido: str, etiqueta: str, limite_s: float) -> dict:
    t0 = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"pedido": pedido, "etiqueta": etiqueta})
    respuesta.raise_for_status()
    job_id = respuesta.json()["job_id"]
    ms_post = (time.perf_counter() - t0) * 1000

    while time.perf_counter() - t0 < limite_s:
        await asyncio.sleep(2)
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] in ESTADOS_FINALES:
            break
    else:
        job = {"estado": "TIMEOUT", "error": f"No termino en {limite_s:.0f} s"}

    return {
        "job_id": job_id,
        "pedido": pedido,
        "estado": job["estado"],
        "error": job.get("error"),
        "ms_respuesta_post": round(ms_post, 1),
        # Medido por el worker: solo la ejecucion del grafo.
        "duracion_grafo_s": job.get("duracion_segundos"),
        # Medido por el cliente: desde el POST hasta ver el estado final
        # (incluye la espera en la cola y la granularidad del polling, 2 s).
        "duracion_cliente_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------
# 3: metricas desde Phoenix
# --------------------------------------------------------------------------

CONSULTA_PROYECTOS = "{ projects { edges { node { id name } } } }"
CONSULTA_METRICAS = """
query($id: ID!, $filtro: String) {
  node(id: $id) {
    ... on Project {
      costSummary(filterCondition: $filtro) {
        prompt { cost tokens } completion { cost tokens } total { cost tokens }
      }
      p50: latencyMsQuantile(probability: 0.5, filterCondition: $filtro)
      p95: latencyMsQuantile(probability: 0.95, filterCondition: $filtro)
    }
  }
}"""


async def metricas_phoenix(url: str, proyecto: str, etiqueta: str, esperadas: int) -> dict | None:
    """Consulta la API de Phoenix (la misma que usa su dashboard)."""
    filtro = f"metadata['etiqueta'] == '{etiqueta}'"
    async with httpx.AsyncClient(base_url=url, timeout=15) as phoenix:

        async def gql(consulta: str, variables: dict | None = None) -> dict:
            r = await phoenix.post("/graphql", json={"query": consulta, "variables": variables or {}})
            r.raise_for_status()
            return r.json()["data"]

        # El worker manda los spans en lotes (cada ~5 s): puede que las trazas
        # (y hasta el proyecto, la primera vez) tarden unos segundos en aparecer.
        trazas, datos = 0, None
        for _ in range(15):
            edges = (await gql(CONSULTA_PROYECTOS))["projects"]["edges"]
            proyectos = {e["node"]["name"]: e["node"]["id"] for e in edges}
            if proyecto in proyectos:
                trazas = await contar_trazas(phoenix, proyectos[proyecto], filtro)
                if trazas >= esperadas:
                    break
            await asyncio.sleep(2)

        if proyecto not in proyectos:
            print(f"  Phoenix no tiene el proyecto '{proyecto}'. Proyectos: {list(proyectos)}")
            return None
        if trazas < esperadas:
            print(f"  Atencion: Phoenix tiene {trazas} de {esperadas} trazas con esta etiqueta.")
        datos = (await gql(CONSULTA_METRICAS, {"id": proyectos[proyecto], "filtro": filtro}))["node"]

    total = datos["costSummary"]["total"]
    return {
        "filtro_en_phoenix": filtro,
        "trazas": trazas,
        "costo_total_usd": total["cost"],
        "costo_por_ejecucion_usd": (total["cost"] / trazas) if total["cost"] and trazas else None,
        "tokens_entrada": datos["costSummary"]["prompt"]["tokens"],
        "tokens_salida": datos["costSummary"]["completion"]["tokens"],
        "latencia_p50_s": datos["p50"] / 1000 if datos["p50"] else None,
        "latencia_p95_s": datos["p95"] / 1000 if datos["p95"] else None,
    }


async def contar_trazas(phoenix: httpx.AsyncClient, proyecto_id: str, filtro: str) -> int:
    """Cuantas trazas (spans raiz) tienen la etiqueta: una por ejecucion del grafo."""
    consulta = """query($id: ID!, $filtro: String) { node(id: $id) { ... on Project {
        spans(first: 100, filterCondition: $filtro) { edges { node { id } } } } } }"""
    filtro_raiz = f"{filtro} and parent_id is None"
    r = await phoenix.post("/graphql", json={"query": consulta, "variables": {"id": proyecto_id, "filtro": filtro_raiz}})
    r.raise_for_status()
    return len(r.json()["data"]["node"]["spans"]["edges"])


# --------------------------------------------------------------------------
# Salida
# --------------------------------------------------------------------------


def percentil(valores: list[float], p: float) -> float:
    """Percentil con interpolacion lineal (el mismo criterio que numpy por defecto)."""
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * p
    abajo = int(k)
    arriba = min(abajo + 1, len(ordenados) - 1)
    return ordenados[abajo] + (ordenados[arriba] - ordenados[abajo]) * (k - abajo)


def mostrar(resultados: list[dict], metricas: dict | None, total_s: float) -> None:
    print("\n" + "=" * 78)
    print(f"{'job':<10}{'estado':<12}{'POST (ms)':>10}{'grafo (s)':>11}{'cliente (s)':>13}  pedido")
    print("-" * 78)
    for r in resultados:
        print(f"{r['job_id'][:8]:<10}{r['estado']:<12}{r['ms_respuesta_post']:>10}"
              f"{(r['duracion_grafo_s'] or 0):>11}{r['duracion_cliente_s']:>13}  {r['pedido'][:30]}...")
    print("-" * 78)
    print(f"Tiempo total de la corrida: {total_s:.1f} s "
          f"(si fueran en fila seria ~{sum(r['duracion_grafo_s'] or 0 for r in resultados):.0f} s)")

    duraciones = [r["duracion_grafo_s"] for r in resultados if r["estado"] == "DONE" and r["duracion_grafo_s"]]
    if duraciones:
        print(f"Segun el worker  -> p50 {statistics.median(duraciones):.1f} s | "
              f"p95 {percentil(duraciones, 0.95):.1f} s | max {max(duraciones):.1f} s")

    if metricas:
        print(f"Segun Phoenix    -> p50 {metricas['latencia_p50_s'] or 0:.1f} s | "
              f"p95 {metricas['latencia_p95_s'] or 0:.1f} s | {metricas['trazas']} trazas")
        if metricas["costo_total_usd"] is not None:
            print(f"Costo (Phoenix)  -> total US$ {metricas['costo_total_usd']:.4f} | "
                  f"por ejecucion US$ {metricas['costo_por_ejecucion_usd']:.4f} | "
                  f"tokens {metricas['tokens_entrada']:.0f} entrada / {metricas['tokens_salida']:.0f} salida")
        print(f"\nPara las capturas, filtra en Phoenix con:\n  {metricas['filtro_en_phoenix']}")
    print("Nota: con 5 muestras, el p95 queda practicamente en el valor maximo.")
    print("=" * 78)


async def main() -> None:
    argumentos = argparse.ArgumentParser(description="Prueba de carga de la API (PE7)")
    argumentos.add_argument("--api", default="http://localhost:8000")
    argumentos.add_argument("--phoenix", default="http://localhost:6006")
    argumentos.add_argument("--proyecto", default="orquestador-menus")
    argumentos.add_argument("--limite", type=float, default=600, help="segundos maximos por job")
    args = argumentos.parse_args()

    etiqueta = f"carga-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"Lanzando {len(PEDIDOS)} pedidos concurrentes | etiqueta: {etiqueta}")

    inicio = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.api, timeout=30) as cliente:
        resultados = await asyncio.gather(
            *[lanzar_y_esperar(cliente, pedido, etiqueta, args.limite) for pedido in PEDIDOS]
        )
    total_s = time.perf_counter() - inicio

    print("Jobs terminados. Consultando metricas en Phoenix...")
    try:
        metricas = await metricas_phoenix(args.phoenix, args.proyecto, etiqueta, len(PEDIDOS))
    except httpx.HTTPError as error:
        print(f"  No se pudo consultar Phoenix ({error}). Se muestran solo las metricas del worker.")
        metricas = None

    mostrar(resultados, metricas, total_s)

    salida = RAIZ / "logs" / f"carga_{etiqueta}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(
        {"etiqueta": etiqueta, "total_s": round(total_s, 1), "jobs": resultados, "phoenix": metricas},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"Resultados guardados en: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    asyncio.run(main())
