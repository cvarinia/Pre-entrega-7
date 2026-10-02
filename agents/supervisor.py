"""Nodo Supervisor: el router inteligente y controlador de flujo.

El Supervisor COORDINA, no ejecuta: nunca busca recetas ni hace cuentas. En
cada ronda:
    1. Lee el pedido original y los aportes de los especialistas (resumidos).
    2. Recibe los chequeos determinísticos de `validacion.py`.
    3. Aplica su rubrica y decide, con salida estructurada (Literal), si
       interviene el investigador, el analista, o si es momento de FINALIZAR.
    4. Si manda a un especialista, le escribe una instruccion autocontenida
       (el especialista no ve la conversacion: solo esa instruccion).

Condicion de parada (contra el "Supervisor infinito"):
    - Criterio de suficiencia: la rubrica + los chequeos automaticos.
    - Contador de rondas: al llegar a `max_rondas_supervisor`, el CODIGO
      fuerza el cierre sin preguntarle al LLM y marca la tarea como no
      completada, para que la respuesta final lo diga (revision humana).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.validacion import chequeos_a_texto, chequeos_automaticos, todo_ok
from comun.config import Configuracion
from comun.mensajes import texto_de_mensaje
from comun.reintentos import con_reintentos
from state import Aporte, DecisionSupervisor, EstadoOrquestador, RegistroDecision

PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo de cocina con dos especialistas:

- "investigador": encuentra recetas en la base de datos (por etiqueta, por
  ingrediente) y entrega una seleccion de ids. No calcula nada.
- "analista": procesa las recetas seleccionadas: verifica restricciones
  alimentarias, calcula calorias y arma la lista de compras. No busca recetas.

Tu trabajo es COORDINAR, no ejecutar. Nunca respondas vos el pedido: decidi
quien interviene ahora o si es momento de FINALIZAR.

Rubrica de suficiencia (FINALIZAR solo si se cumple todo):
1. La seleccion del investigador responde al pedido completo: cantidad de
   recetas, restricciones alimentarias y tipo de plato (por ejemplo, si se
   piden cenas, un postre o un budin NO sirve).
2. El analista proceso exactamente esa seleccion (chequeo de consistencia OK).
3. Estan todos los resultados que pidio la persona (calorias, lista de compras, etc.).
4. Ningun chequeo automatico figura como FALLA.

Reglas de ruteo:
- Si no hay seleccion, o la seleccion no cumple el punto 1 -> "investigador",
  con una instruccion que diga exactamente que corregir (que receta sacar y por que).
- Si hay una seleccion correcta pero el analisis falta, quedo viejo o tiene
  una falla -> "analista".
- Si el investigador corrigio la seleccion, el analisis anterior ya no vale:
  hay que volver a pasar por el analista.
- Las instrucciones tienen que ser AUTOCONTENIDAS: el especialista no ve esta
  conversacion. Inclui cantidad de recetas, comensales, restricciones y tipo de plato.

Envio de la lista de compras (solo al FINALIZAR):
- Marca solicita_envio=true UNICAMENTE si la persona pidio que se ENVIE o MANDE
  la lista (al almacen, al super, etc.). Si solo pidio la lista para verla,
  solicita_envio=false. Vos no enviás nada: el envio lo hace otro paso del
  sistema y requiere aprobacion humana.
"""


def _resumen_de_aporte(aporte: Aporte) -> str:
    """Resume un aporte para el Supervisor sin volcarle toda la evidencia cruda."""
    lineas = [f"[ronda {aporte.ronda}] {aporte.agente}: {aporte.resumen}"]
    if aporte.receta_ids:
        lineas.append(f"  ids: {aporte.receta_ids}")
    for ev in aporte.evidencia:
        if ev.error:
            lineas.append(f"  herramienta {ev.herramienta} devolvio error: {ev.error}")
        elif ev.herramienta == "entregar_seleccion":
            lineas.append(f"  seleccion entregada: {json.dumps(ev.resultado.get('seleccion'), ensure_ascii=False)}")
        elif ev.herramienta == "verificar_etiquetas":
            lineas.append(f"  verificacion: {json.dumps(ev.resultado, ensure_ascii=False)}")
        elif ev.herramienta == "calcular_nutricion_menu":
            r = ev.resultado
            lineas.append(f"  calorias: total={r.get('calorias_totales_menu')} "
                          f"por_comensal={r.get('calorias_por_comensal')} comensales={r.get('comensales')}")
        elif ev.herramienta == "consolidar_lista_compras":
            lineas.append(f"  lista de compras: {ev.resultado.get('cantidad_de_items')} items")
    return "\n".join(lineas)


def _pedido_original(estado: EstadoOrquestador) -> str:
    for mensaje in estado["messages"]:
        if isinstance(mensaje, HumanMessage):
            return texto_de_mensaje(mensaje)
    return ""


def crear_nodo_supervisor(modelo: BaseChatModel, config: Configuracion):
    """Devuelve la funcion-nodo del Supervisor."""
    decisor = modelo.with_structured_output(DecisionSupervisor)

    async def nodo_supervisor(estado: EstadoOrquestador) -> dict[str, Any]:
        ronda = estado.get("ronda", 0) + 1
        aportes = estado.get("aportes", [])
        chequeos = chequeos_automaticos(aportes)

        # --- Condicion de parada dura: la decide el codigo, no el LLM -------
        if ronda > config.max_rondas_supervisor:
            registro = RegistroDecision(
                ronda=ronda,
                evaluacion=f"Se alcanzo el maximo de {config.max_rondas_supervisor} rondas sin cumplir la rubrica.",
                siguiente="FINALIZAR",
                instruccion="",
                forzada=True,
            )
            print(f"  [supervisor] ronda {ronda}: limite alcanzado -> FINALIZAR (forzado)", flush=True)
            return {
                "ronda": ronda,
                "next_agent": "FINALIZAR",
                "instruccion_actual": "",
                "decisiones": [registro],
                "task_completed": False,
                "motivo_cierre": registro.evaluacion + " Requiere revision humana.",
                "solicita_envio": False,  # un cierre forzado nunca dispara el envio
                "messages": [AIMessage(content=f"[ronda {ronda}] FINALIZAR (forzado por limite)", name="supervisor")],
            }

        # --- Decision del LLM con la rubrica ---------------------------------
        contexto = (
            f"PEDIDO DE LA PERSONA:\n{_pedido_original(estado)}\n\n"
            f"RONDA ACTUAL: {ronda} de {config.max_rondas_supervisor}\n\n"
            "APORTES DE LOS ESPECIALISTAS (del mas viejo al mas nuevo):\n"
            + ("\n".join(_resumen_de_aporte(a) for a in aportes) or "(todavia ninguno)")
            + "\n\nCHEQUEOS AUTOMATICOS (calculados por codigo, son confiables):\n"
            + chequeos_a_texto(chequeos)
            + "\n\n¿Quien interviene ahora, o es momento de FINALIZAR?"
        )

        async def decidir() -> DecisionSupervisor:
            return await asyncio.wait_for(
                decisor.ainvoke([SystemMessage(PROMPT_SUPERVISOR), HumanMessage(contexto)]),
                timeout=config.timeout_modelo,
            )

        decision = await con_reintentos(decidir)
        registro = RegistroDecision(ronda=ronda, **decision.model_dump())
        print(f"  [supervisor] ronda {ronda}: -> {decision.siguiente} | {decision.evaluacion}", flush=True)

        actualizacion: dict[str, Any] = {
            "ronda": ronda,
            "next_agent": decision.siguiente,
            "instruccion_actual": decision.instruccion,
            "decisiones": [registro],
            "messages": [AIMessage(
                content=f"[ronda {ronda}] -> {decision.siguiente}. {decision.evaluacion}",
                name="supervisor",
            )],
        }

        if decision.siguiente == "FINALIZAR":
            # "Confiar pero verificar": el LLM puede decidir cerrar, pero la
            # tarea solo cuenta como completa si los chequeos lo respaldan.
            completa = todo_ok(chequeos)
            actualizacion["task_completed"] = completa
            # PE7: el pedido de envio viaja al estado; el grafo decide despues
            # si pasa por la aprobacion humana (solo si ademas la tarea esta completa).
            actualizacion["solicita_envio"] = decision.solicita_envio
            actualizacion["motivo_cierre"] = (
                "Rubrica cumplida y chequeos automaticos OK." if completa else
                "El Supervisor cerro con chequeos pendientes o fallidos: "
                + "; ".join(c.detalle for c in chequeos if c.ok is False)
                + " Requiere revision humana."
            )

        return actualizacion

    return nodo_supervisor
