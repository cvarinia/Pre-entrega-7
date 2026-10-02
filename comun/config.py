"""Configuracion central del proyecto (reutilizada de la pre-entrega 5).

Un unico lugar donde se leen las variables de entorno y se resuelven las rutas.
Ningun otro modulo llama a os.getenv(): si manana cambia el modelo, el
proveedor o un limite, se toca un solo archivo.

Novedades respecto de la PE5: desaparece el checkpointer (esta entrega no
necesita memoria entre sesiones) y aparecen los limites propios de un sistema
multi-agente: cuantas rondas puede coordinar el Supervisor y cuantos pasos
internos puede dar cada especialista.

Novedades de la PE7: vuelve el checkpointer (ahora en Redis) y se agregan la
URL de Redis y cuantos jobs puede ejecutar el worker al mismo tiempo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Raiz del repositorio (config.py esta en <raiz>/comun/config.py)
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent

PROVEEDORES_VALIDOS: tuple[str, ...] = ("gemini", "anthropic")

VARIABLE_DE_CLAVE: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Version fijada a proposito: los alias tipo "latest" cambian solos y
# volverian irreproducible la entrega.
MODELO_POR_DEFECTO: dict[str, str] = {
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-haiku-4-5-20251001",
}

MARCADORES_DE_EJEMPLO: tuple[str, ...] = ("pega_aca_tu_clave", "tu_clave", "sk-ant-xxx")


@dataclass(frozen=True)
class Configuracion:
    """Parametros de ejecucion. Inmutable: se arma una vez y nadie la toca."""

    proveedor: str
    api_key: str
    modelo: str
    temperatura: float
    # Rondas maximas del Supervisor: la condicion de parada contra el
    # "Supervisor infinito" que advierte la consigna.
    max_rondas_supervisor: int
    # Pasos internos maximos de cada especialista (su propio ciclo ReAct).
    limite_pasos_especialista: int
    # Techo global del grafo, como red de seguridad de ultima instancia.
    limite_recursion_grafo: int
    timeout_modelo: float
    ruta_recetas: Path
    ruta_logs: Path
    # --- PE7 ---------------------------------------------------------------
    # Donde viven la cola de jobs, su estado y los checkpoints de LangGraph.
    redis_url: str = "redis://localhost:6379/0"
    # Cuantos grafos ejecuta el worker a la vez. Con 1, las 5 peticiones de la
    # prueba de carga correrian en fila y no habria concurrencia real.
    max_trabajos_concurrentes: int = 5
    # Observabilidad: a donde se mandan las trazas (Arize Phoenix) y con que
    # nombre de proyecto aparecen en el dashboard.
    observabilidad: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    proyecto_phoenix: str = "orquestador-menus"

    @classmethod
    def desde_entorno(cls) -> "Configuracion":
        """Construye la configuracion leyendo el .env. Falla temprano y claro."""
        load_dotenv(RAIZ_PROYECTO / ".env")

        proveedor = os.getenv("PROVEEDOR", "gemini").strip().lower()
        if proveedor not in PROVEEDORES_VALIDOS:
            raise RuntimeError(
                f"PROVEEDOR='{proveedor}' no es valido. Opciones: "
                f"{', '.join(PROVEEDORES_VALIDOS)}."
            )

        variable = VARIABLE_DE_CLAVE[proveedor]
        api_key = os.getenv(variable, "").strip().strip("\"'")

        if not api_key:
            raise RuntimeError(
                f"Falta {variable} para el proveedor '{proveedor}'. Copia "
                ".env.example a .env y completa esa variable."
            )
        if api_key in MARCADORES_DE_EJEMPLO or len(api_key) < 20:
            raise RuntimeError(
                f"{variable} parece no ser una clave real. Abri el .env y "
                "reemplaza el valor por tu clave, sin comillas ni espacios."
            )
        if proveedor == "anthropic" and not api_key.startswith("sk-ant-"):
            raise RuntimeError("ANTHROPIC_API_KEY deberia empezar con 'sk-ant-'.")

        return cls(
            proveedor=proveedor,
            api_key=api_key,
            modelo=os.getenv("MODELO_LLM", "").strip() or MODELO_POR_DEFECTO[proveedor],
            temperatura=float(os.getenv("TEMPERATURA", "0")),
            max_rondas_supervisor=int(os.getenv("MAX_RONDAS_SUPERVISOR", "6")),
            limite_pasos_especialista=int(os.getenv("LIMITE_PASOS_ESPECIALISTA", "12")),
            limite_recursion_grafo=int(os.getenv("LIMITE_RECURSION_GRAFO", "30")),
            timeout_modelo=float(os.getenv("TIMEOUT_MODELO", "90")),
            ruta_recetas=RAIZ_PROYECTO / "data" / "recetas.json",
            ruta_logs=RAIZ_PROYECTO / "logs",
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0").strip(),
            max_trabajos_concurrentes=int(os.getenv("MAX_TRABAJOS_CONCURRENTES", "5")),
            observabilidad=os.getenv("OBSERVABILIDAD", "true").strip().lower() in ("1", "true", "si", "sí"),
            phoenix_endpoint=os.getenv(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
            ).strip(),
            proyecto_phoenix=os.getenv("PHOENIX_PROJECT_NAME", "orquestador-menus").strip(),
        )

    @classmethod
    def para_pruebas(cls) -> "Configuracion":
        """Configuracion sin API key real, para los tests con modelo simulado."""
        return cls(
            proveedor="gemini",
            api_key="clave-de-prueba-suficientemente-larga",
            modelo="modelo-simulado",
            temperatura=0.0,
            max_rondas_supervisor=6,
            limite_pasos_especialista=12,
            limite_recursion_grafo=30,
            timeout_modelo=90.0,
            ruta_recetas=RAIZ_PROYECTO / "data" / "recetas.json",
            ruta_logs=RAIZ_PROYECTO / "logs",
        )
