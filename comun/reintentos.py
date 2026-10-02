"""Reintentos automaticos ante errores temporales del proveedor.

No todos los errores de una API son definitivos. Un 429 (cuota por minuto
agotada) o un 503 (servicio momentaneamente saturado) se resuelven solos
esperando unos segundos. Sin esta capa, un pico de trafico de Google corta una
corrida entera del agente a mitad de camino.

Ojo con la diferencia: la cuota POR MINUTO se recupera esperando; la cuota
DIARIA del free tier, no. Por eso el reintento tiene un techo de espera y una
cantidad limitada de intentos: si el problema es la cuota del dia, falla rapido
con el mensaje del proveedor en vez de dormir para siempre.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

# Marcas de error que suelen ser transitorias.
SENIALES_TEMPORALES: tuple[str, ...] = (
    "429",
    "RESOURCE_EXHAUSTED",
    "rate limit",
    "503",
    "UNAVAILABLE",
    "overloaded",
    "timeout",  # incluye asyncio.TimeoutError: la llamada tardo demasiado
    "deadline",
)

# El proveedor suele sugerir cuanto esperar: "retryDelay': '43s'" o "retry in 43.0s".
_PATRON_DEMORA = re.compile(r"retry(?:Delay|\s+in)['\":\s]+([0-9]+(?:\.[0-9]+)?)s", re.I)


def es_error_temporal(error: BaseException) -> bool:
    """True si el error tiene pinta de resolverse esperando un rato."""
    texto = f"{type(error).__name__}: {error}"
    return any(senial.lower() in texto.lower() for senial in SENIALES_TEMPORALES)


def demora_sugerida(error: BaseException) -> float | None:
    """Extrae del mensaje los segundos que el proveedor pide esperar, si los dice."""
    coincidencia = _PATRON_DEMORA.search(str(error))
    return float(coincidencia.group(1)) if coincidencia else None


async def con_reintentos(
    operacion: Callable[[], Awaitable[T]],
    intentos: int = 3,
    espera_base: float = 2.0,
    espera_maxima: float = 65.0,
) -> T:
    """Ejecuta una corrutina reintentando ante errores temporales.

    Espera lo que pida el proveedor si lo indica; si no, usa backoff exponencial
    (2s, 4s, 8s...). Los errores que no son temporales se propagan enseguida:
    no tiene sentido reintentar una API key invalida.

    Args:
        operacion: Funcion sin argumentos que devuelve la corrutina a ejecutar.
        intentos: Cantidad total de ejecuciones, incluida la primera.
        espera_base: Segundos de la primera espera del backoff.
        espera_maxima: Techo de espera. Si el proveedor pide mas que esto, se
            considera que el problema no se resuelve en esta corrida y falla.

    Returns:
        Lo que devuelva la operacion.
    """
    ultimo_error: BaseException | None = None

    for intento in range(1, intentos + 1):
        try:
            return await operacion()
        except Exception as error:  # noqa: BLE001 - se re-lanza abajo si no aplica
            ultimo_error = error

            if intento == intentos or not es_error_temporal(error):
                raise

            # Ojo: `or` no sirve aca, porque una demora sugerida de 0 segundos es
            # un valor valido y falsy a la vez.
            sugerida = demora_sugerida(error)
            espera = sugerida if sugerida is not None else espera_base * (2 ** (intento - 1))
            if espera > espera_maxima:
                # Tipico de la cuota diaria agotada: esperar no sirve.
                raise

            print(
                f"   [reintento {intento}/{intentos - 1}] error temporal del "
                f"proveedor, esperando {espera:.0f}s..."
            )
            await asyncio.sleep(espera)

    assert ultimo_error is not None  # inalcanzable: el for siempre sale por return/raise
    raise ultimo_error
