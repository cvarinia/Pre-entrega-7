"""Lectura de las trazas de la prueba de carga, nodo por nodo (PE7).

Responde las preguntas de la consigna con los datos de Phoenix:
    - ¿Donde se concentra la latencia?
    - ¿Que nodo del grafo consume mas tokens (y costo)?

Como lo calcula:
    - Trae de Phoenix todos los spans de las trazas con la etiqueta de la carga.
    - Los NODOS del grafo son los hijos directos del span raiz "orquestador"
      (supervisor, investigador, analista, sintesis, ...). Un nodo puede
      ejecutarse varias veces por traza (el Supervisor, por ejemplo, 3 veces).
    - Cada llamada al LLM se le asigna al nodo del que cuelga, subiendo por el
      arbol. Asi se suman los tokens y el costo de cada nodo.

Uso:
    python scripts/lectura_trazas.py                     (usa la ultima carga de logs/)
    python scripts/lectura_trazas.py --etiqueta carga-20261002-134715
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import httpx

RAIZ = Path(__file__).resolve().parent.parent

CONSULTA_SPANS = """
query($id: ID!, $filtro: String, $despues: String) {
  node(id: $id) { ... on Project {
    spans(first: 500, after: $despues, filterCondition: $filtro) {
      pageInfo { hasNextPage endCursor }
      edges { node {
        name spanId parentId spanKind latencyMs
        tokenCountPrompt tokenCountCompletion
        costSummary { total { cost } }
      } }
    }
  } }
}"""


def ultima_etiqueta() -> str:
    cargas = sorted((RAIZ / "logs").glob("carga_*.json"))
    if not cargas:
        raise SystemExit("No hay cargas en logs/. Corre primero: python scripts/carga.py")
    return json.loads(cargas[-1].read_text(encoding="utf-8"))["etiqueta"]


def traer_spans(url: str, proyecto: str, etiqueta: str) -> list[dict]:
    with httpx.Client(base_url=url, timeout=30) as phoenix:
        def gql(consulta: str, variables: dict | None = None) -> dict:
            r = phoenix.post("/graphql", json={"query": consulta, "variables": variables or {}})
            r.raise_for_status()
            cuerpo = r.json()
            if cuerpo.get("errors"):
                raise SystemExit(f"Phoenix respondio con error: {cuerpo['errors']}")
            return cuerpo["data"]

        edges = gql("{ projects { edges { node { id name } } } }")["projects"]["edges"]
        proyectos = {e["node"]["name"]: e["node"]["id"] for e in edges}
        if proyecto not in proyectos:
            raise SystemExit(f"Phoenix no tiene el proyecto '{proyecto}'.")

        spans, despues = [], None
        while True:  # paginado: una traza tiene decenas de spans
            pagina = gql(CONSULTA_SPANS, {
                "id": proyectos[proyecto],
                "filtro": f"metadata['etiqueta'] == '{etiqueta}'",
                "despues": despues,
            })["node"]["spans"]
            spans += [e["node"] for e in pagina["edges"]]
            if not pagina["pageInfo"]["hasNextPage"]:
                return spans
            despues = pagina["pageInfo"]["endCursor"]


def analizar(spans: list[dict]) -> dict:
    por_id = {s["spanId"]: s for s in spans}
    raices = {s["spanId"] for s in spans if s["parentId"] is None}
    nodos = {s["spanId"]: s for s in spans if s["parentId"] in raices}

    def nodo_de(span: dict) -> dict | None:
        """Sube por el arbol hasta encontrar el nodo del grafo del que cuelga."""
        actual = span
        while actual is not None and actual["spanId"] not in nodos:
            actual = por_id.get(actual["parentId"])
        return actual

    resumen: dict[str, dict] = defaultdict(lambda: {
        "ejecuciones": 0, "latencia_ms": 0.0, "llamadas_llm": 0,
        "tokens_entrada": 0.0, "tokens_salida": 0.0, "costo_usd": 0.0,
    })
    for nodo in nodos.values():
        fila = resumen[nodo["name"]]
        fila["ejecuciones"] += 1
        fila["latencia_ms"] += nodo["latencyMs"] or 0

    for span in spans:
        if (span["spanKind"] or "").lower() != "llm":
            continue
        nodo = nodo_de(span)
        if nodo is None:
            continue
        fila = resumen[nodo["name"]]
        fila["llamadas_llm"] += 1
        fila["tokens_entrada"] += span["tokenCountPrompt"] or 0
        fila["tokens_salida"] += span["tokenCountCompletion"] or 0
        costo = (span.get("costSummary") or {}).get("total", {}).get("cost")
        fila["costo_usd"] += costo or 0

    latencia_total = sum(por_id[r]["latencyMs"] or 0 for r in raices)
    return {"trazas": len(raices), "latencia_total_ms": latencia_total, "nodos": dict(resumen)}


def mostrar(etiqueta: str, analisis: dict) -> None:
    n = analisis["trazas"] or 1
    total_ms = analisis["latencia_total_ms"] or 1
    nodos = analisis["nodos"]
    costo_total = sum(f["costo_usd"] for f in nodos.values()) or 1
    tokens_total = sum(f["tokens_entrada"] + f["tokens_salida"] for f in nodos.values()) or 1

    print(f"\nLectura de las trazas | etiqueta: {etiqueta} | {analisis['trazas']} trazas")
    print("=" * 96)
    print(f"{'nodo':<18}{'veces/traza':>12}{'seg/traza':>11}{'% tiempo':>10}"
          f"{'llamadas LLM':>14}{'tokens/traza':>14}{'% tokens':>10}{'% costo':>9}")
    print("-" * 96)
    for nombre, f in sorted(nodos.items(), key=lambda kv: -kv[1]["latencia_ms"]):
        tokens = f["tokens_entrada"] + f["tokens_salida"]
        print(f"{nombre:<18}{f['ejecuciones'] / n:>12.1f}{f['latencia_ms'] / n / 1000:>11.1f}"
              f"{100 * f['latencia_ms'] / total_ms:>9.0f}%{f['llamadas_llm'] / n:>14.1f}"
              f"{tokens / n:>14.0f}{100 * tokens / tokens_total:>9.0f}%"
              f"{100 * f['costo_usd'] / costo_total:>8.0f}%")
    print("-" * 96)
    print(f"Duracion media de una traza: {total_ms / n / 1000:.1f} s")
    print("=" * 96)


def main() -> None:
    argumentos = argparse.ArgumentParser(description="Lectura de trazas por nodo (PE7)")
    argumentos.add_argument("--etiqueta", default=None)
    argumentos.add_argument("--phoenix", default="http://localhost:6006")
    argumentos.add_argument("--proyecto", default="orquestador-menus")
    args = argumentos.parse_args()

    etiqueta = args.etiqueta or ultima_etiqueta()
    analisis = analizar(traer_spans(args.phoenix, args.proyecto, etiqueta))
    mostrar(etiqueta, analisis)

    salida = RAIZ / "logs" / f"lectura_{etiqueta}.json"
    salida.write_text(json.dumps({"etiqueta": etiqueta, **analisis}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado en: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
