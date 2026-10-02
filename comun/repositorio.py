"""Capa de acceso a datos: la 'base de datos' simulada de recetas.

Reutilizada de la pre-entrega 5. Unico agregado: `buscar_por_etiqueta`, que
necesita el Agente de Investigacion para pedidos del tipo "sin gluten" o
"vegetariano".

Esta capa NO sabe que existe un LLM ni una herramienta de LangChain. Solo sabe
leer recetas. Gracias a eso, el dia que las recetas vengan de Postgres, de una
API o de un vector store, se reescribe unicamente este archivo y ni las tools ni
el grafo se enteran.

Todo es asincronico: la lectura del JSON se delega a un thread con
`asyncio.to_thread` para no bloquear el event loop, igual que haria un driver de
base de datos real.
"""

from __future__ import annotations

import asyncio
import json
import unicodedata
from pathlib import Path
from typing import Any, TypedDict


class Receta(TypedDict):
    """Forma de una receta en la base simulada."""

    id: str
    nombre: str
    ingredientes: list[str]
    porciones_base: int
    calorias_por_porcion: int
    tiempo_minutos: int
    etiquetas: list[str]


def _normalizar(texto: str) -> str:
    """Pasa a minusculas y saca acentos, para que 'Limon' matchee con 'limón'."""
    sin_acentos = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sin_acentos if not unicodedata.combining(c)).strip()


class RepositorioRecetas:
    """Lee el JSON de recetas y ofrece consultas asincronicas sobre el."""

    def __init__(self, ruta: Path) -> None:
        self._ruta = ruta
        self._recetas: list[Receta] | None = None
        self._candado = asyncio.Lock()

    async def _cargar(self) -> list[Receta]:
        """Carga perezosa y con candado: el archivo se lee una sola vez por proceso."""
        if self._recetas is not None:
            return self._recetas

        async with self._candado:
            if self._recetas is None:  # doble chequeo por si otra corrutina ya cargo
                contenido: dict[str, Any] = await asyncio.to_thread(
                    lambda: json.loads(self._ruta.read_text(encoding="utf-8"))
                )
                self._recetas = contenido["recetas"]
        return self._recetas

    async def buscar_por_ingrediente(
        self, ingrediente: str, limite: int = 5
    ) -> list[Receta]:
        """Devuelve las recetas cuyo listado de ingredientes contiene el termino buscado."""
        recetas = await self._cargar()
        objetivo = _normalizar(ingrediente)
        encontradas = [
            receta
            for receta in recetas
            if any(objetivo in _normalizar(ing) for ing in receta["ingredientes"])
        ]
        return encontradas[:limite]

    async def buscar_por_etiqueta(self, etiqueta: str, limite: int = 10) -> list[Receta]:
        """Devuelve las recetas que tienen una etiqueta alimentaria determinada.

        Regla de dominio: toda receta vegana es tambien vegetariana, asi que
        buscar "vegetariana" incluye las etiquetadas como "vegana".
        """
        recetas = await self._cargar()
        objetivo = _normalizar(etiqueta)
        aceptadas = {objetivo}
        if objetivo == "vegetariana":
            aceptadas.add("vegana")
        encontradas = [
            receta
            for receta in recetas
            if aceptadas & {_normalizar(e) for e in receta["etiquetas"]}
        ]
        return encontradas[:limite]

    async def etiquetas_disponibles(self) -> list[str]:
        """Lista las etiquetas unicas que existen en la base."""
        recetas = await self._cargar()
        return sorted({e for receta in recetas for e in receta["etiquetas"]})

    async def obtener(self, receta_id: str) -> Receta | None:
        """Devuelve una receta por id, o None si no existe."""
        recetas = await self._cargar()
        objetivo = _normalizar(receta_id)
        for receta in recetas:
            if _normalizar(receta["id"]) == objetivo:
                return receta
        return None

    async def ids_disponibles(self) -> list[str]:
        """Lista todos los ids, util para armar mensajes de error informativos."""
        recetas = await self._cargar()
        return [receta["id"] for receta in recetas]

    async def ingredientes_disponibles(self) -> list[str]:
        """Lista los ingredientes unicos que existen en la base."""
        recetas = await self._cargar()
        unicos = {ing for receta in recetas for ing in receta["ingredientes"]}
        return sorted(unicos)


# Instancia unica del repositorio, creada la primera vez que se pide.
# Las tools la consumen a traves de `obtener_repositorio()`.
_repositorio: RepositorioRecetas | None = None


def obtener_repositorio(ruta: Path | None = None) -> RepositorioRecetas:
    """Devuelve el repositorio compartido, creandolo en la primera llamada."""
    global _repositorio
    if _repositorio is None:
        if ruta is None:
            from comun.config import RAIZ_PROYECTO

            ruta = RAIZ_PROYECTO / "data" / "recetas.json"
        _repositorio = RepositorioRecetas(ruta)
    return _repositorio
