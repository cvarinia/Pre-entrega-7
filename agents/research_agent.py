"""Agente especialista de Busqueda/Investigacion.

Responsabilidad unica: ENCONTRAR recetas en la base que cumplan el pedido.
No hace cuentas, no arma listas de compras, no redacta la respuesta final.

Herramientas acotadas a su rol (ninguna calcula nada):
    - buscar_recetas_por_etiqueta     (sin gluten, vegetariana, ...)
    - buscar_recetas_por_ingrediente  (reutilizada de la PE5)
    - obtener_receta                  (reutilizada de la PE5)
    - entregar_seleccion              (su "entrega formal" al Supervisor)

`entregar_seleccion` es la pieza clave de la proveniencia: la seleccion del
investigador no se lee del texto libre del modelo (que podria alucinar un id),
sino de los argumentos de esta herramienta, que ademas valida contra la base
que cada id exista.
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

NOMBRE = "investigador"

PROMPT_INVESTIGADOR = """Sos el Agente de Investigacion de un equipo de cocina.
Tu unico trabajo es ENCONTRAR recetas en la base de datos que cumplan la
instruccion que recibis. No hagas calculos de calorias ni listas de compras:
de eso se ocupa otro agente.

Como trabajas:
- Nunca inventes recetas ni ids. Todo lo que afirmes tiene que salir de una herramienta.
- Para restricciones alimentarias (sin gluten, sin lacteos, vegetariana, vegana)
  usa buscar_recetas_por_etiqueta. Para ingredientes, buscar_recetas_por_ingrediente.
- buscar_recetas_por_etiqueta ya trae las fichas completas: elegi directamente
  a partir de ese resultado. Usa obtener_receta solo si te falta un dato.
- Cada llamada a una herramienta cuesta tiempo y cuota: resolvelo con la menor
  cantidad de llamadas posible.
- Si la instruccion pide un tipo de plato (por ejemplo "cenas" o "platos
  principales"), descarta postres, budines y acompañamientos.
- Si una herramienta devuelve "error", lee la "sugerencia" y reintenta distinto.
- Cuando tengas la seleccion, llama OBLIGATORIAMENTE a entregar_seleccion con
  los ids elegidos y una justificacion breve. Despues responde en una o dos
  lineas en español rioplatense resumiendo que elegiste.
"""


# --------------------------------------------------------------------------
# Herramientas
# --------------------------------------------------------------------------


@tool
async def buscar_recetas_por_etiqueta(etiqueta: str, max_resultados: int = 10) -> dict[str, Any]:
    """Busca recetas que tengan una etiqueta alimentaria.

    Usala para pedidos con restricciones: "sin gluten", "sin lacteos",
    "vegetariana", "vegana". Buscar "vegetariana" incluye tambien las veganas.

    Devuelve las fichas completas (ingredientes, etiquetas, calorias), asi que
    en general NO hace falta llamar despues a obtener_receta.

    Args:
        etiqueta: Una de: "sin_gluten", "contiene_gluten", "sin_lacteos",
            "contiene_lacteos", "vegetariana", "vegana".
        max_resultados: Cantidad maxima de recetas a devolver.
    """
    repo = obtener_repositorio()
    normalizada = etiqueta.strip().lower().replace(" ", "_")
    encontradas = await repo.buscar_por_etiqueta(normalizada, limite=max_resultados)
    if not encontradas:
        return {
            "error": f"No hay recetas con la etiqueta '{etiqueta}'.",
            "sugerencia": "Reintenta con una de las etiquetas disponibles.",
            "etiquetas_disponibles": await repo.etiquetas_disponibles(),
        }
    return {
        "cantidad": len(encontradas),
        # Fichas completas: ahorra una llamada al modelo por cada receta que el
        # agente tendria que abrir con obtener_receta (cuida la cuota del free tier).
        "recetas": [dict(r) for r in encontradas],
    }


@tool
async def buscar_recetas_por_ingrediente(ingrediente: str, max_resultados: int = 5) -> dict[str, Any]:
    """Busca recetas que contengan un ingrediente determinado.

    Usala cuando el pedido menciona un alimento concreto ("algo con papa",
    "tengo lentejas"). Un solo ingrediente por llamada, en singular.
    Devuelve solo id, nombre, tiempo y porciones base.

    Args:
        ingrediente: Nombre del ingrediente, por ejemplo "pollo" o "papa".
        max_resultados: Cantidad maxima de recetas a devolver.
    """
    repo = obtener_repositorio()
    encontradas = await repo.buscar_por_ingrediente(ingrediente, limite=max_resultados)
    if not encontradas:
        return {
            "error": f"No hay ninguna receta con el ingrediente '{ingrediente}'.",
            "sugerencia": "Reintenta con uno de los ingredientes que existen en la base.",
            "ingredientes_disponibles": await repo.ingredientes_disponibles(),
        }
    return {
        "cantidad": len(encontradas),
        "recetas": [
            {"id": r["id"], "nombre": r["nombre"], "tiempo_minutos": r["tiempo_minutos"]}
            for r in encontradas
        ],
    }


@tool
async def obtener_receta(receta_id: str) -> dict[str, Any]:
    """Devuelve la ficha completa de una receta: ingredientes, porciones, calorias y etiquetas.

    Las etiquetas son la unica fuente confiable para saber si una receta cumple
    una restriccion. No lo deduzcas del nombre.

    Args:
        receta_id: Identificador exacto, con el formato "r001".
    """
    repo = obtener_repositorio()
    receta = await repo.obtener(receta_id)
    if receta is None:
        return {
            "error": f"No existe ninguna receta con el id '{receta_id}'.",
            "sugerencia": "Los ids tienen formato 'r001'. Busca primero por etiqueta o ingrediente.",
            "ids_de_ejemplo": (await repo.ids_disponibles())[:5],
        }
    return dict(receta)


@tool
async def entregar_seleccion(receta_ids: list[str], justificacion: str) -> dict[str, Any]:
    """Entrega formalmente al Supervisor las recetas elegidas. Llamala UNA vez, al final.

    Valida que cada id exista en la base. Si alguno no existe, devuelve error y
    tenes que corregir la seleccion y volver a llamarla.

    Args:
        receta_ids: Ids de las recetas elegidas, por ejemplo ["r009", "r017"].
        justificacion: Una frase explicando por que cumplen el pedido.
    """
    repo = obtener_repositorio()
    fichas, inexistentes = [], []
    for receta_id in receta_ids:
        receta = await repo.obtener(receta_id)
        if receta is None:
            inexistentes.append(receta_id)
        else:
            fichas.append({"id": receta["id"], "nombre": receta["nombre"], "etiquetas": receta["etiquetas"]})

    if inexistentes or not fichas:
        return {
            "error": f"Ids inexistentes o seleccion vacia: {inexistentes or receta_ids}.",
            "sugerencia": "Verifica los ids con obtener_receta y volve a entregar.",
        }
    return {"aceptada": True, "seleccion": fichas, "justificacion": justificacion}


HERRAMIENTAS_INVESTIGADOR = [
    buscar_recetas_por_etiqueta,
    buscar_recetas_por_ingrediente,
    obtener_receta,
    entregar_seleccion,
]


# --------------------------------------------------------------------------
# Nodo del grafo
# --------------------------------------------------------------------------


def seleccion_entregada(aporte_evidencia: list) -> list[str]:
    """Ids de la ULTIMA entrega aceptada (si el agente corrigio, vale la ultima)."""
    for ev in reversed(aporte_evidencia):
        if ev.herramienta == "entregar_seleccion" and ev.error is None and isinstance(ev.resultado, dict):
            return [f["id"] for f in ev.resultado.get("seleccion", [])]
    return []


def crear_nodo_investigador(modelo: BaseChatModel, config: Configuracion):
    """Arma el sub-agente ReAct y devuelve la funcion-nodo para el grafo."""
    agente = create_react_agent(
        modelo, HERRAMIENTAS_INVESTIGADOR, prompt=PROMPT_INVESTIGADOR, name=NOMBRE
    )

    async def nodo_investigador(estado: EstadoOrquestador) -> dict[str, Any]:
        # Contexto minimo: SOLO la instruccion del Supervisor. No ve el
        # historial global ni lo que hicieron otros agentes.
        instruccion = estado["instruccion_actual"]
        print(f"  [investigador] {instruccion}", flush=True)

        try:
            mensajes = await ejecutar_especialista(agente, instruccion, config)
            evidencia = extraer_evidencia(mensajes)
            resumen = texto_final(mensajes)
        except (GraphRecursionError, asyncio.TimeoutError) as error:
            # No dejar el estado "en el limbo": se registra un aporte fallido
            # y el Supervisor decide que hacer con el.
            evidencia = []
            resumen = f"No pude completar la busqueda: {type(error).__name__}."

        ids = seleccion_entregada(evidencia)
        if not ids:
            resumen += " (ATENCION: no se registro ninguna seleccion con entregar_seleccion)"

        aporte = Aporte(
            agente=NOMBRE,
            ronda=estado["ronda"],
            instruccion=instruccion,
            resumen=resumen,
            receta_ids=ids,
            evidencia=evidencia,
        )
        return {
            "aportes": [aporte],
            "messages": [AIMessage(content=resumen, name=NOMBRE)],
        }

    return nodo_investigador
