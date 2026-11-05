# Práctica de Laboratorio: Motor Agéntico ReAct

**Asignatura:** Agentes Inteligentes (VIII Ciclo)

**Universidad:** Universidad Nacional del Santa

## Integrantes

- Mendoza Chávez Raúl
- Álvarez Paredes José

## Descripción

Este proyecto implementa, a bajo nivel y sin frameworks como LangChain o CrewAI, un motor agéntico basado en el paradigma **ReAct** (*Reasoning + Acting*). El agente utiliza un modelo de lenguaje de la API de Gemini para razonar, invocar herramientas de Python y corregirse ante errores.

El ciclo implementado es:

`Thought → Action → Action Input → Observation → Final Answer`

## Funcionalidades

**Ejercicio 1: Autocorrección (Self-Healing)**

- Catálogo de herramientas: `consultar_inventario` y `calcular_descuento`, con validación de entrada.
- Los errores (`KeyError`, `ValueError`, `TypeError` o herramienta inexistente) se capturan y se devuelven al modelo como `Observation: [ERROR DE EJECUCIÓN]`, para que corrija su llamada.
- Uso de `stop_sequences=["Observation:"]` para que el modelo se detenga y ceda el control al entorno.

**Ejercicio 2: Parseo robusto y Failsafe**

- Parseo con expresiones regulares tolerante a cercados Markdown, espacios, comillas y variaciones de mayúsculas.
- Inyección del timestamp del sistema en el *System Prompt* (`[ESTADO_SISTEMA]`).
- Límite de `max_pasos = 5` con respuesta de degradación `[FAILSAFE ACTIVADO]`.

## Estructura del proyecto

```
.
├── practica_semana05_react.py   # Código principal
├── requirements.txt             # Dependencias
├── .env.example                 # Ejemplo de configuración de la clave
├── .gitignore
└── README.md
```

## Requisitos

- Python 3.9 o superior
- Una clave de la API de Gemini (se obtiene en [Google AI Studio](https://aistudio.google.com/))

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración de la clave de API

1. Copia el archivo `.env.example` y renómbralo como `.env`.
2. Escribe tu clave en ese archivo, sin comillas:

```
GEMINI_API_KEY=tu_clave_aqui
```

Como alternativa, se puede definir la variable de entorno `GEMINI_API_KEY` en la terminal.

## Ejecución

```bash
python practica_semana05_react.py
```

El programa procesa tres consultas de prueba:

1. Consulta de precio con descuento (flujo exitoso con autocorrección).
2. Consulta de un producto inexistente (`KeyError`).
3. Descuento fuera de rango (`ValueError`).

## Ejemplo de salida

```
=== PROCESANDO PREGUNTA: '¿Cuántas unidades hay del producto PROD-999?' ===

--- PASO 1 de 5 ---
Thought: Necesito averiguar la cantidad de unidades disponibles...
Action: consultar_inventario
Action Input: PROD-999
Observation: [ERROR DE EJECUCIÓN] KeyError: El producto con ID 'PROD-999' no existe...

--- PASO 2 de 5 ---
Final Answer: No es posible consultar el producto 'PROD-999' ya que no existe...
```

## Modelo utilizado

`gemini-3.1-flash-lite`, configurado en la constante `MODELO` del archivo principal. Si Google retira el modelo, basta con cambiar ese valor.