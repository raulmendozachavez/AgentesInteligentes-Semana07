import os
import re
import json
import math
import time
import inspect
import datetime
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# Carga la clave desde .env si python-dotenv está instalado
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODELO = "gemini-3.1-flash-lite"
STOP_SEQUENCES = ["Observation:"]

# ==========================================
# 1. CATÁLOGO DE HERRAMIENTAS Y SIMULACIÓN
# ==========================================
BD_INVENTARIO = {
    "PROD-001": {"nombre": "Laptop Gamer", "stock": 5, "precio": 1200.0},
    "PROD-002": {"nombre": "Monitor 4K", "stock": 0, "precio": 350.0},
}


def consultar_inventario(producto_id: str) -> str:
    """Consulta el stock y precio de un producto por su ID."""
    if not isinstance(producto_id, str):
        raise TypeError("'producto_id' debe ser texto, por ejemplo 'PROD-001'.")
    producto_id = producto_id.strip().upper()
    if producto_id not in BD_INVENTARIO:
        raise KeyError(
            f"El producto con ID '{producto_id}' no existe en la base de datos. "
            f"IDs válidos: {list(BD_INVENTARIO.keys())}"
        )
    info = BD_INVENTARIO[producto_id]
    return f"Producto: {info['nombre']} | Stock: {info['stock']} | Precio: ${info['precio']}"


def calcular_descuento(precio: float, porcentaje: float) -> str:
    """Calcula el precio final aplicando un porcentaje de descuento."""
    try:
        p = float(precio)
        d = float(porcentaje)
    except (ValueError, TypeError):
        raise ValueError("Los argumentos 'precio' y 'porcentaje' deben ser números válidos.")
    if not (math.isfinite(p) and math.isfinite(d)):
        raise ValueError("Los argumentos 'precio' y 'porcentaje' deben ser números finitos.")
    if p < 0 or d < 0 or d > 100:
        raise ValueError("El precio debe ser positivo y el descuento debe estar entre 0 y 100.")
    precio_final = p * (1 - d / 100)
    return f"Precio original: ${p:.2f} | Descuento: {d}% | Precio Final: ${precio_final:.2f}"


HERRAMIENTAS_DISPONIBLES = {
    "consultar_inventario": consultar_inventario,
    "calcular_descuento": calcular_descuento,
}

# ==========================================
# 2. CLIENTE DE LA API DE GEMINI
# ==========================================


class ClienteGemini:
    def __init__(self, modelo: str = MODELO, reintentos: int = 3):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Define la variable de entorno GEMINI_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        self.modelo = modelo
        self.reintentos = reintentos

    def generar(self, contexto: str, stop_sequences: List[str]) -> str:
        config = types.GenerateContentConfig(
            stop_sequences=stop_sequences,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            http_options=types.HttpOptions(timeout=60_000),
        )
        for intento in range(1, self.reintentos + 1):
            try:
                respuesta = self.client.models.generate_content(
                    model=self.modelo, contents=contexto, config=config
                )
                texto = respuesta.text or ""
                if not texto.strip():
                    print("[Aviso] Respuesta vacía del modelo.")
                return texto
            except Exception as e:
                codigo = getattr(e, "code", None)
                # Errores de clave o de modelo no se reintentan
                if codigo in (400, 401, 403, 404) or intento == self.reintentos:
                    raise
                espera = 2 ** intento
                print(f"[Aviso] Error {codigo}. Reintento en {espera}s ({intento}/{self.reintentos}).")
                time.sleep(espera)
        return ""


# ==========================================
# 3. MOTOR DE PARSEO Y AUTOCORRECCIÓN
# ==========================================
# Fin de un campo: otra etiqueta o el final del texto
_FIN = r"(?=\n\s*(?:Thought|Action\s*Input|Action|Observation|Final\s+Answer)\s*:|\Z)"
_BANDERAS = re.IGNORECASE | re.DOTALL
_RUIDO = "`'\" \n\t"


def parsear_argumentos(args_str: str) -> Tuple[list, dict]:
    """Convierte el Action Input en argumentos posicionales y con nombre."""
    s = (args_str or "").strip().strip("`").strip()
    if not s:
        return [], {}
    if s.startswith("{"):
        try:
            return [], dict(json.loads(s))
        except json.JSONDecodeError:
            raise ValueError("El Action Input parece JSON pero es inválido.")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    args, kwargs = [], {}
    for parte in (p.strip() for p in s.split(",")):
        if not parte:
            continue
        if "=" in parte:
            clave, valor = parte.split("=", 1)
            kwargs[clave.strip()] = valor.strip().strip(_RUIDO)
        else:
            args.append(parte.strip(_RUIDO))
    return args, kwargs


class MotorReActEstudiante:
    def __init__(self, max_pasos: int = 5, llm: Optional[ClienteGemini] = None):
        self.max_pasos = max_pasos
        self.llm = llm if llm is not None else ClienteGemini()
        self.historial_contexto: str = ""

    def obtener_reloj_sistema(self) -> str:
        """TAREA 1 (Ej 2): Obtener timestamp actual del sistema."""
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def construir_system_prompt(self) -> str:
        timestamp = self.obtener_reloj_sistema()
        return f"""Eres un Agente Autónomo ReAct. Tu deber es resolver consultas respondiendo
estrictamente en este formato:

Thought: <tu razonamiento sobre qué hacer>
Action: <nombre de la herramienta a usar>
Action Input: <argumentos para la herramienta>
Observation: <resultado que el entorno te devolverá>

Reglas:
- Emite UNA sola acción por turno y detente. Nunca escribas la Observation tú mismo.
- Si una Observation indica un error, razona sobre él y corrige tu llamada.
- Cuando tengas la respuesta final, responde exactamente con:
Final Answer: <respuesta clara y directa al usuario>

Herramientas disponibles:
- consultar_inventario: Argumento (producto_id: str)
- calcular_descuento: Argumentos (precio, porcentaje)

[ESTADO_SISTEMA]: Timestamp actual = {timestamp}
"""

    def parsear_salida_llm(self, texto_salida: str) -> Dict[str, Optional[str]]:
        """TAREA 2 (Ej 2): Parseo mediante RegEx tolerante a Markdown y variaciones."""
        res = {"thought": None, "action": None, "action_input": None, "final_answer": None}
        if not texto_salida:
            return res

        # Elimina cercados de código y negritas
        texto = re.sub(r"```[a-zA-Z]*", "", texto_salida).replace("```", "")
        texto = texto.replace("**", "")

        m_final = re.search(r"\bFinal\s+Answer\s*:\s*(.*)", texto, _BANDERAS)
        m_action = re.search(r"\bAction\s*:\s*(.*?)" + _FIN, texto, _BANDERAS)
        m_input = re.search(r"\bAction\s*Input\s*:\s*(.*?)" + _FIN, texto, _BANDERAS)
        m_thought = re.search(r"\bThought\s*:\s*(.*?)" + _FIN, texto, _BANDERAS)

        if m_thought:
            res["thought"] = m_thought.group(1).strip()

        # Si hay Action y Final Answer, gana el que aparece primero
        usar_accion = m_action and (not m_final or m_action.start() < m_final.start())
        if usar_accion:
            nombre = re.match(r"[A-Za-z0-9_]+", m_action.group(1).strip(_RUIDO))
            res["action"] = nombre.group(0) if nombre else None
            if m_input:
                res["action_input"] = m_input.group(1).strip(_RUIDO)
        elif m_final:
            res["final_answer"] = m_final.group(1).strip(_RUIDO)
        return res

    def ejecutar_herramienta_con_self_healing(self, nombre_herramienta: str, args_str: str) -> str:
        """TAREA 3 (Ej 1): Ejecutar herramienta y capturar excepciones para Self-Healing."""
        sufijo = "Por favor corrige los argumentos o intenta otra herramienta."
        if nombre_herramienta not in HERRAMIENTAS_DISPONIBLES:
            return (f"Observation: [ERROR DE EJECUCIÓN] La herramienta '{nombre_herramienta}' "
                    f"no existe. Herramientas válidas: {list(HERRAMIENTAS_DISPONIBLES.keys())}. {sufijo}")

        func = HERRAMIENTAS_DISPONIBLES[nombre_herramienta]
        try:
            args, kwargs = parsear_argumentos(args_str)
            resultado = func(*args, **kwargs)
            return f"Observation: [ÉXITO] {resultado}"
        except TypeError as e:
            firma = f"{nombre_herramienta}{inspect.signature(func)}"
            return (f"Observation: [ERROR DE EJECUCIÓN] TypeError: {e}. "
                    f"Firma esperada: {firma}. {sufijo}")
        except Exception as e:
            mensaje = str(e.args[0] if e.args else e).rstrip(". ")
            return f"Observation: [ERROR DE EJECUCIÓN] {type(e).__name__}: {mensaje}. {sufijo}"

    def ejecutar_bucle_react(self, pregunta_usuario: str) -> str:
        """Simulador del bucle ReAct con control de pasos y Failsafe."""
        print(f"\n=== PROCESANDO PREGUNTA: '{pregunta_usuario}' ===")
        self.historial_contexto = (
            self.construir_system_prompt() + f"\nUsuario: {pregunta_usuario}\n"
        )

        for paso in range(1, self.max_pasos + 1):
            print(f"\n--- PASO {paso} de {self.max_pasos} ---")

            try:
                salida = self.llm.generar(self.historial_contexto, STOP_SEQUENCES)
            except Exception as e:
                detalle = getattr(e, "message", None) or str(e)
                respuesta = f"Final Answer: [ERROR DE API] {type(e).__name__}: {detalle}"
                print(respuesta)
                return respuesta

            # Corta cualquier Observation inventada por el modelo
            salida = salida.split("Observation:")[0].rstrip()
            print(salida)
            self.historial_contexto += salida + "\n"

            parsed = self.parsear_salida_llm(salida)

            if parsed["final_answer"]:
                return f"Final Answer: {parsed['final_answer']}"

            if parsed["action"]:
                observacion = self.ejecutar_herramienta_con_self_healing(
                    parsed["action"], parsed["action_input"] or ""
                )
            else:
                # Formato inválido: el modelo recibe el error para corregirse
                observacion = ("Observation: [ERROR DE EJECUCIÓN] No se detectó Action ni "
                               "Final Answer. Responde con el formato Thought/Action/Action Input "
                               "o con Final Answer. Por favor corrige tu respuesta.")

            print(observacion)
            self.historial_contexto += observacion + "\n"

        # Failsafe: se alcanzó el límite de pasos
        respuesta = (f"Final Answer: [FAILSAFE ACTIVADO] Se ha alcanzado el límite máximo de "
                     f"{self.max_pasos} pasos sin obtener una respuesta final. "
                     f"Reformula la consulta o intenta nuevamente.")
        print(f"\n{respuesta}")
        return respuesta


# ==========================================
# 4. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    agente = MotorReActEstudiante(max_pasos=5)
    preguntas = [
        "¿Cuánto cuesta la Laptop Gamer con 10% de descuento?",
        "¿Cuántas unidades hay del producto PROD-999?",
        "Aplica 150% de descuento a la Laptop Gamer.",
    ]
    for pregunta in preguntas:
        respuesta = agente.ejecutar_bucle_react(pregunta)
        print(f"\n>>> {respuesta}")