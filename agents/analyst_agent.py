"""Agente especialista de Analisis/Computo.

Responsabilidad unica: PROCESAR las recetas que eligio el investigador.
No busca recetas nuevas: si no le alcanzan los datos, lo dice.

Principio de diseño: el LLM decide QUE calculo pedir, pero los numeros los
produce Python. Ninguna caloria ni ninguna lista de compras sale de la
"cabeza" del modelo: salen de estas herramientas determinísticas.

Herramientas acotadas a su rol (ninguna busca recetas):
    - verificar_etiquetas          (validacion de esquema/restricciones)
    - calcular_nutricion_menu      (reutilizada de la PE5)
    - consolidar_lista_compras     (union de ingredientes con escala)
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from agents.base import ejecutar_especialista, extraer_evidencia, texto_final
from comun.config import Configuracion
from comun.repositorio import obtener_repositorio
from state import Aporte, EstadoOrquestador

NOMBRE = "analista"

# Herramientas cuyos argumentos `receta_ids` cuentan como "recetas analizadas".
HERRAMIENTAS_DE_CALCULO = ("calcular_nutricion_menu", "consolidar_lista_compras")

PROMPT_ANALISTA = """Sos el Agente de Analisis de un equipo de cocina.
Recibis una instruccion y los ids de las recetas que eligio el Agente de
Investigacion. Tu trabajo es PROCESAR esas recetas con tus herramientas.

Como trabajas:
- Trabaja SOLO con los ids que te pasan. No agregues ni cambies recetas.
- Nunca hagas cuentas de cabeza: todo numero sale de una herramienta.
- Si la instruccion menciona una restriccion alimentaria, verificala con
  verificar_etiquetas.
- Para calorias usa calcular_nutricion_menu; para la lista de compras,
  consolidar_lista_compras. Usa la cantidad de comensales que pide la instruccion.
- Estas herramientas son independientes entre si: pedi TODAS las que
  necesites JUNTAS, en un mismo turno, en lugar de una por vez.
- Si una herramienta devuelve "error", lee la "sugerencia" y reintenta.
- Al terminar, responde en pocas lineas en español rioplatense con los numeros
  concretos que obtuviste. Si detectaste un problema (por ejemplo una receta
  que no cumple la restriccion), decilo explicitamente.
"""


# --------------------------------------------------------------------------
# Herramientas
# --------------------------------------------------------------------------


@tool
async def verificar_etiquetas(receta_ids: list[str], etiqueta_requerida: str) -> dict[str, Any]:
    """Verifica que TODAS las recetas cumplan una etiqueta alimentaria.

    Es una validacion determinística: compara contra las etiquetas de la base,
    sin interpretar nombres. Una receta "vegana" cuenta como "vegetariana".

    Args:
        receta_ids: Ids a verificar, por ejemplo ["r009", "r017"].
        etiqueta_requerida: Por ejemplo "sin_gluten" o "vegetariana".
    """
    repo = obtener_repositorio()
    requerida = etiqueta_requerida.strip().lower().replace(" ", "_")
    aceptadas = {requerida, "vegana"} if requerida == "vegetariana" else {requerida}

    cumplen, no_cumplen, inexistentes = [], [], []
    for receta_id in receta_ids:
        receta = await repo.obtener(receta_id)
        if receta is None:
            inexistentes.append(receta_id)
        elif aceptadas & set(receta["etiquetas"]):
            cumplen.append(receta_id)
        else:
            no_cumplen.append({"id": receta_id, "nombre": receta["nombre"], "etiquetas": receta["etiquetas"]})

    return {
        "etiqueta_requerida": requerida,
        "todas_cumplen": not no_cumplen and not inexistentes,
        "cumplen": cumplen,
        "no_cumplen": no_cumplen,
        "inexistentes": inexistentes,
    }


@tool
async def calcular_nutricion_menu(receta_ids: list[str], comensales: int = 4) -> dict[str, Any]:
    """Calcula las calorias totales de un menu y cuantas le tocan a cada comensal.

    Suma una porcion de cada receta por comensal. Informa ademas el factor de
    escala de cada receta respecto de sus porciones base.

    Args:
        receta_ids: Ids de las recetas del menu.
        comensales: Cantidad de personas. Por defecto 4.
    """
    repo = obtener_repositorio()
    if comensales < 1:
        return {"error": "La cantidad de comensales tiene que ser al menos 1.",
                "sugerencia": "Usa la cantidad que indica la instruccion."}

    detalle, advertencias, total = [], [], 0
    for receta_id in receta_ids:
        receta = await repo.obtener(receta_id)
        if receta is None:
            advertencias.append(f"El id '{receta_id}' no existe y fue ignorado.")
            continue
        calorias_grupo = receta["calorias_por_porcion"] * comensales
        total += calorias_grupo
        detalle.append({
            "id": receta["id"],
            "nombre": receta["nombre"],
            "calorias_por_porcion": receta["calorias_por_porcion"],
            "factor_de_escala": round(comensales / receta["porciones_base"], 2),
            "calorias_para_el_grupo": calorias_grupo,
        })

    if not detalle:
        return {"error": "Ninguno de los ids existe en la base.", "advertencias": advertencias,
                "sugerencia": "Usa los ids que te paso el Supervisor."}
    return {
        "comensales": comensales,
        "detalle": detalle,
        "calorias_totales_menu": total,
        "calorias_por_comensal": total // comensales,
        "advertencias": advertencias,
    }


@tool
async def consolidar_lista_compras(receta_ids: list[str], comensales: int = 4) -> dict[str, Any]:
    """Arma una lista de compras unica a partir de varias recetas.

    Une los ingredientes repetidos e indica en que recetas se usa cada uno. La
    base no tiene cantidades por ingrediente, asi que se informa el factor de
    escala de cada receta (comensales / porciones base) para ajustar al comprar.

    Args:
        receta_ids: Ids de las recetas del menu.
        comensales: Cantidad de personas. Por defecto 4.
    """
    repo = obtener_repositorio()
    usos: dict[str, list[str]] = {}
    escalas, inexistentes = {}, []
    for receta_id in receta_ids:
        receta = await repo.obtener(receta_id)
        if receta is None:
            inexistentes.append(receta_id)
            continue
        escalas[receta["nombre"]] = round(comensales / receta["porciones_base"], 2)
        for ingrediente in receta["ingredientes"]:
            usos.setdefault(ingrediente, []).append(receta["nombre"])

    if not usos:
        return {"error": "Ninguno de los ids existe en la base.", "inexistentes": inexistentes,
                "sugerencia": "Usa los ids que te paso el Supervisor."}
    return {
        "comensales": comensales,
        "cantidad_de_items": len(usos),
        "items": [{"ingrediente": ing, "se_usa_en": recetas} for ing, recetas in sorted(usos.items())],
        "factor_de_escala_por_receta": escalas,
        "inexistentes": inexistentes,
    }


HERRAMIENTAS_ANALISTA = [verificar_etiquetas, calcular_nutricion_menu, consolidar_lista_compras]


# --------------------------------------------------------------------------
# Nodo del grafo
# --------------------------------------------------------------------------


def ultima_seleccion(estado: EstadoOrquestador) -> list[str]:
    """Ids de la seleccion mas reciente del investigador (proveniencia)."""
    for aporte in reversed(estado.get("aportes", [])):
        if aporte.agente == "investigador" and aporte.receta_ids:
            return aporte.receta_ids
    return []


def recetas_analizadas(evidencia: list) -> list[str]:
    """Ids que el analista efectivamente proceso (segun sus llamadas exitosas)."""
    ids: list[str] = []
    for ev in evidencia:
        if ev.herramienta in HERRAMIENTAS_DE_CALCULO and ev.error is None:
            for receta_id in ev.argumentos.get("receta_ids", []):
                if receta_id not in ids:
                    ids.append(receta_id)
    return ids


def crear_nodo_analista(modelo: BaseChatModel, config: Configuracion):
    """Arma el sub-agente ReAct y devuelve la funcion-nodo para el grafo."""
    agente = create_react_agent(
        modelo, HERRAMIENTAS_ANALISTA, prompt=PROMPT_ANALISTA, name=NOMBRE
    )

    async def nodo_analista(estado: EstadoOrquestador) -> dict[str, Any]:
        # Contexto minimo: la instruccion del Supervisor + los ids verificados
        # del investigador. Nada del historial de conversacion.
        seleccion = ultima_seleccion(estado)
        instruccion = (
            f"{estado['instruccion_actual']}\n\n"
            f"Ids seleccionados por el Agente de Investigacion: "
            f"{seleccion if seleccion else 'NINGUNO TODAVIA (informalo y no calcules)'}"
        )
        print(f"  [analista] {estado['instruccion_actual']} | ids={seleccion}", flush=True)

        try:
            mensajes = await ejecutar_especialista(agente, instruccion, config)
            evidencia = extraer_evidencia(mensajes)
            resumen = texto_final(mensajes)
        except (GraphRecursionError, asyncio.TimeoutError) as error:
            evidencia = []
            resumen = f"No pude completar el analisis: {type(error).__name__}."

        aporte = Aporte(
            agente=NOMBRE,
            ronda=estado["ronda"],
            instruccion=estado["instruccion_actual"],
            resumen=resumen,
            receta_ids=recetas_analizadas(evidencia),
            evidencia=evidencia,
        )
        return {
            "aportes": [aporte],
            "messages": [AIMessage(content=resumen, name=NOMBRE)],
        }

    return nodo_analista
