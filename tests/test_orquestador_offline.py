"""Pruebas del orquestador SIN consumir la API (modelo simulado).

Verifican el cableado: ruteo del Supervisor, ida y vuelta con los
especialistas, refinamiento, proveniencia, deteccion de conflictos y la
condicion de parada. Las herramientas y los chequeos son REALES (leen la base
de recetas); lo unico simulado son las decisiones del LLM.

Si estas pruebas pasan y algo falla con Gemini, el problema esta en el prompt
o en el modelo, no en la arquitectura.

Uso:  python -m tests.test_orquestador_offline
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from itertools import count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage  # noqa: E402

from comun.config import Configuracion  # noqa: E402
from comun.modelo_simulado import ModeloSimulado  # noqa: E402
from graph import construir_grafo, estado_inicial  # noqa: E402
from state import DecisionSupervisor  # noqa: E402

_ids = count(1)


def pedir(herramienta: str, **args) -> AIMessage:
    """AIMessage que pide una herramienta, como haria el modelo real."""
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": args, "id": f"c{next(_ids)}"}])


def decidir(siguiente: str, instruccion: str = "", evaluacion: str = "-") -> DecisionSupervisor:
    return DecisionSupervisor(siguiente=siguiente, instruccion=instruccion, evaluacion=evaluacion)


def investigacion(ids: list[str]) -> list[AIMessage]:
    return [
        pedir("buscar_recetas_por_etiqueta", etiqueta="vegetariana"),
        pedir("entregar_seleccion", receta_ids=ids, justificacion="Cumplen el pedido."),
        AIMessage(content=f"Elegi {ids}."),
    ]


def analisis(ids: list[str]) -> list[AIMessage]:
    return [
        pedir("verificar_etiquetas", receta_ids=ids, etiqueta_requerida="vegetariana"),
        pedir("calcular_nutricion_menu", receta_ids=ids, comensales=6),
        pedir("consolidar_lista_compras", receta_ids=ids, comensales=6),
        AIMessage(content="Calorias y lista de compras listas."),
    ]


async def correr(modelo: ModeloSimulado, config: Configuracion) -> dict:
    grafo = construir_grafo(config, modelo=modelo)
    return await grafo.ainvoke(estado_inicial("3 cenas vegetarianas para 6"), {"recursion_limit": 40})


async def prueba_flujo_con_refinamiento() -> None:
    """Investigador elige un budin -> Supervisor pide corregir -> analista -> fin."""
    modelo = ModeloSimulado(
        respuestas=[
            *investigacion(["r008", "r009", "r012"]),   # r012 es un budin: no es cena
            *investigacion(["r008", "r009", "r018"]),   # seleccion corregida
            *analisis(["r008", "r009", "r018"]),
            AIMessage(content="Respuesta final simulada."),  # sintesis
        ],
        decisiones=[
            decidir("investigador", "Busca 3 cenas vegetarianas para 6 personas."),
            decidir("investigador", "Reemplaza r012 (budin, no es cena) por un plato principal."),
            decidir("analista", "Verifica 'vegetariana' y calcula calorias y compras para 6."),
            decidir("FINALIZAR"),
        ],
    )
    final = await correr(modelo, Configuracion.para_pruebas())

    recorrido = [d.siguiente for d in final["decisiones"]]
    assert recorrido == ["investigador", "investigador", "analista", "FINALIZAR"], recorrido
    assert final["task_completed"] is True, final["motivo_cierre"]

    investigaciones = [a for a in final["aportes"] if a.agente == "investigador"]
    assert investigaciones[0].receta_ids == ["r008", "r009", "r012"]   # proveniencia: la historia no se borra
    assert investigaciones[1].receta_ids == ["r008", "r009", "r018"]

    analista = next(a for a in final["aportes"] if a.agente == "analista")
    nutricion = next(e for e in analista.evidencia if e.herramienta == "calcular_nutricion_menu")
    # (450 + 330 + 290) kcal por porcion -> 1070 por comensal
    assert nutricion.resultado["calorias_por_comensal"] == 1070, nutricion.resultado
    assert final["messages"][-1].name == "sintesis"
    print(f"OK  flujo con refinamiento: {' -> '.join(recorrido)}")


async def prueba_conflicto_entre_agentes() -> None:
    """El analista procesa otras recetas: el chequeo lo detecta y la tarea NO cuenta como completa."""
    modelo = ModeloSimulado(
        respuestas=[
            *investigacion(["r008", "r009", "r018"]),
            *analisis(["r008", "r009", "r001"]),   # r001 no la eligio el investigador
            AIMessage(content="Respuesta final simulada."),
        ],
        decisiones=[
            decidir("investigador", "Busca 3 cenas vegetarianas."),
            decidir("analista", "Calcula calorias y compras."),
            decidir("FINALIZAR"),   # el LLM "se equivoca" y quiere cerrar igual
        ],
    )
    final = await correr(modelo, Configuracion.para_pruebas())

    assert final["task_completed"] is False
    assert "CONFLICTO" in final["motivo_cierre"], final["motivo_cierre"]
    print("OK  conflicto detectado por proveniencia -> tarea marcada como incompleta")


async def prueba_limite_de_rondas() -> None:
    """Si el Supervisor nunca queda conforme, el codigo corta en el limite."""
    config = replace(Configuracion.para_pruebas(), max_rondas_supervisor=2)
    modelo = ModeloSimulado(
        respuestas=[
            *investigacion(["r012"]),
            *investigacion(["r016"]),
            AIMessage(content="Respuesta final simulada."),
        ],
        decisiones=[
            decidir("investigador", "Busca cenas."),
            decidir("investigador", "Otra vez, eso no es una cena."),
            # no hay tercera decision: la ronda 3 la resuelve el codigo
        ],
    )
    final = await correr(modelo, config)

    ultima = final["decisiones"][-1]
    assert ultima.forzada is True and ultima.siguiente == "FINALIZAR"
    assert final["task_completed"] is False
    print("OK  limite de rondas: cierre forzado por codigo y marcado para revision humana")


async def main() -> None:
    await prueba_flujo_con_refinamiento()
    await prueba_conflicto_entre_agentes()
    await prueba_limite_de_rondas()
    print("\nTodas las pruebas pasaron.")


if __name__ == "__main__":
    asyncio.run(main())
