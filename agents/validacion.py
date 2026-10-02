"""Validacion determinística del trabajo de los especialistas.

Lo que se puede verificar con certeza se verifica con codigo, no con el LLM:
    - ¿el investigador entrego una seleccion?
    - ¿el analisis es posterior a la ultima seleccion (no quedo viejo)?
    - ¿el analista proceso EXACTAMENTE las recetas del investigador?
      (deteccion de conflictos entre agentes via proveniencia)
    - ¿estan los dos calculos (calorias y lista de compras)?
    - si se verifico una restriccion alimentaria, ¿se cumple?

El resultado se le pasa al Supervisor como parte de su prompt. El LLM queda
para lo que no se puede codificar: juzgar si la seleccion responde al pedido
("¿un budin es una cena?").
"""

from __future__ import annotations

from dataclasses import dataclass

from state import Aporte


@dataclass(frozen=True)
class Chequeo:
    nombre: str
    ok: bool | None  # None = no aplica / no se pudo verificar todavia
    detalle: str
    bloqueante: bool = True


def _ultimo(aportes: list[Aporte], agente: str) -> tuple[int, Aporte] | None:
    for indice in range(len(aportes) - 1, -1, -1):
        if aportes[indice].agente == agente:
            return indice, aportes[indice]
    return None


def _ultima_evidencia(aporte: Aporte, herramienta: str):
    for ev in reversed(aporte.evidencia):
        if ev.herramienta == herramienta and ev.error is None:
            return ev
    return None


def chequeos_automaticos(aportes: list[Aporte]) -> list[Chequeo]:
    """Evalua el estado actual y devuelve la lista de chequeos."""
    chequeos: list[Chequeo] = []

    investigacion = _ultimo(aportes, "investigador")
    if investigacion is None or not investigacion[1].receta_ids:
        chequeos.append(Chequeo("seleccion_entregada", False,
                                "El investigador todavia no entrego una seleccion valida."))
        return chequeos  # sin seleccion, el resto no tiene sentido

    indice_inv, aporte_inv = investigacion
    seleccion = aporte_inv.receta_ids
    chequeos.append(Chequeo("seleccion_entregada", True, f"Seleccion vigente: {seleccion}."))

    analisis = _ultimo(aportes, "analista")
    if analisis is None or analisis[0] < indice_inv:
        chequeos.append(Chequeo("analisis_vigente", False,
                                "No hay analisis posterior a la ultima seleccion."))
        return chequeos

    _, aporte_ana = analisis
    chequeos.append(Chequeo("analisis_vigente", True, "El analisis es posterior a la ultima seleccion."))

    # Conflicto entre agentes: ¿el analista trabajo sobre otras recetas?
    iguales = set(aporte_ana.receta_ids) == set(seleccion)
    chequeos.append(Chequeo(
        "consistencia_entre_agentes",
        iguales,
        "El analista proceso exactamente la seleccion del investigador." if iguales else
        f"CONFLICTO: seleccion={seleccion} pero el analista proceso {aporte_ana.receta_ids}.",
    ))

    faltantes = [h for h in ("calcular_nutricion_menu", "consolidar_lista_compras")
                 if _ultima_evidencia(aporte_ana, h) is None]
    chequeos.append(Chequeo(
        "calculos_completos",
        not faltantes,
        "Estan las calorias y la lista de compras." if not faltantes else
        f"Faltan calculos exitosos de: {faltantes}.",
    ))

    verificacion = _ultima_evidencia(aporte_ana, "verificar_etiquetas")
    if verificacion is None:
        chequeos.append(Chequeo("restriccion_alimentaria", None,
                                "No se verifico ninguna etiqueta (solo importa si el pedido tiene restricciones).",
                                bloqueante=False))
    else:
        resultado = verificacion.resultado or {}
        cumple = bool(resultado.get("todas_cumplen"))
        chequeos.append(Chequeo(
            "restriccion_alimentaria",
            cumple,
            f"Todas cumplen '{resultado.get('etiqueta_requerida')}'." if cumple else
            f"No cumplen '{resultado.get('etiqueta_requerida')}': {resultado.get('no_cumplen')}.",
        ))

    return chequeos


def todo_ok(chequeos: list[Chequeo]) -> bool:
    """True si ningun chequeo bloqueante falla ni quedo pendiente."""
    return all(c.ok is True for c in chequeos if c.bloqueante) and not any(
        c.ok is False for c in chequeos
    )


def chequeos_a_texto(chequeos: list[Chequeo]) -> str:
    marca = {True: "OK", False: "FALLA", None: "N/A"}
    return "\n".join(f"- [{marca[c.ok]}] {c.nombre}: {c.detalle}" for c in chequeos)
