# Agentes_Tributarios_IA
Es un proyecto de aprendizaje personal en el que me he centrado en crear un sistema intelegente de asistencia tributaria para contribuyentes. 
La idea original era ampliar el alcance del proyecto a integrarlo con tools para el acceso a sistemas de gestión de colas de atención telemática o para rellenar o presentar escritos. Tal vez, algún día con algo más de tiempo retome la idea y la potencie con el uso de MCPs.

---
## Índice

1. [¿Cómo funciona el sistema?](#cómo-funciona-el-sistema)
2. [Trabajo previo](#trabajo-previo)
3. [Ejemplo de flujo de uso](#ejemplo-de-flujo-de-uso)
4. [Estructura de archivos](#estructura-de-archivos)
5. [Requisitos computacionales](#requisitos-computacionales)
6. [Modelos utilizados](#modelos-utilizados)

---

## ¿Cómo funciona el sistema?
Se trata de un sistema multiagéntico, basado en modelos Gemini (vía API) y un RAG afinado con reranker, orquestado con **langgraph**.

El funcionamiento es el siguiente: 

1. Un **agente de triaje** recibe las cuestiones e identifica entre un listado de temas cuáles son los que puede incluir la pregunta del usuario. En este primer paso, también diferencia entre cuándo es una simple consulta y cuándo desea llevar a cabo otra acción, en este caso reservar cita con un especialista tributario.
2. La pregunta cruda se pasa un **agente reformulador**. Al contribuyente no se le supone un lenguaje técnico ni legalmente sofisticado. Este agente busca reformular la cuestión para mejorar la precisión del RAG a la hora de encontrar referencias legales que ayuden a dar respuesta a la cuestión. Asimismo, con el historial de preguntas y respuestas anterior ayuda a refinar la pregunta.
3. Conforme a la clasificación que realiza el agente de triaje, el **agente especialista** realiza una consulta en la base de datos vectorial restringiendo la temática de búsqueda con los metadatos que se generaron duraron el proceso de almacenamiento de embeddings en ChromaDB. Sobre esta base restringida realiza una búsqueda vectorial con la consulta refinada por el agente refomulador, devolviendo hasta 50 fragmentos que puedan ser relevantes. Estos resultados son evaluados por el reranker, que califica los resultados usando un modelo CrossEncoder con un score conforme a su coherencia con la consulta inicial. De entre esos fragmentos, tan solo se añaden en el contexto del agente especialista aquellos que superan el umbral de 0.6 de score del reranker, para evitar coincidencias residuales.
4. El agente especialista genera una primera respuesta conforme a ese contexto enriquecido con el RAG. Esa respuesta pasa al **agente redactor**, que comprueba la adherencia a la cuestión inicial planteada por el contribuyente y la reformula para hacerlo en un lenguaje accesible para éste.
5. El sistema se ha encapsulado en un microservicio consumible por **API**, a través de FastAPI. Se ha dotado de generación de hilos para recuperar conversaciones previas y poder mantener una conversación sobre preguntas y respuestas previas. La idea inicial era desplegarlo a través de contenedores de Docker.

## Trabajo previo
Previamente se ha alimentado la base de datos vectorial del RAG con información del BOE. Gracias a la librería Docling, lás páginas html se transforman en Markdown. La estructura por bloques del BOE resulta útil para la división en chunks, con los que generar los embeddings que se almacenan en la base de datos vectorial. Para generar los embeddings utilizo el modelo littlejohn-ai/bge-m3-spa-law-qa , disponible bajo autorización de los autores en Hugging Face, que está ajustado con datos y textos legales, por lo que debería ofrecer un mejor rendimiento en la extracción de estas entidades de textos del BOE.

## Ejemplo de flujo de uso
1. El usuario pregunta: 'Se ha muerto mi abuela en Santander, tengo algun beneficio en la casa?'
2. Respuesta agente de triaje: {'tipo': 'consulta', 'contenido': 'GENERAL; ISD; CANTABRIA'}
3. Respuesta agente reformulador: 'En el contexto de la adquisición de un bien inmueble por herencia en Cantabria tras el fallecimiento de un familiar, ¿existen beneficios fiscales aplicables en el Impuesto sobre Sucesiones y Donaciones?'
4. Aquí haría la consulta vectorial, actuaría el reranker y enriquecería la respuesta, pero por longitud de la respuesta saltamos al siguiente paso.
5. Respuesta final agente redactor: Lamento tu pérdida. En relación con la herencia de tu abuela y la vivienda en Santander, existen posibles beneficios fiscales en el Impuesto sobre Sucesiones y Donaciones (ISD) que podrían aplicarse a tu caso.

**Beneficios fiscales aplicables:**

*   **Bonificación por parentesco:** Como nieto, podrías tener derecho a una bonificación del 50% en la cuota del impuesto sobre sucesiones.
*   **Reducción por vivienda habitual:** Si la vivienda era la residencia habitual de tu abuela, se podría aplicar una reducción en la base imponible del impuesto.
*   **Reducción por empresa familiar:** Si la vivienda formaba parte de una empresa familiar o negocio profesional, podría existir una reducción adicional del 95% del valor, siempre que se cumplan ciertos requisitos, como mantener la adquisición durante diez años.

**Aspectos importantes a tener en cuenta:**

*   **Valoración de la vivienda:** La valoración de la vivienda se realizará según las normas del Impuesto sobre Sucesiones y Donaciones.
*   **Impuesto sobre el Patrimonio:** Si tu abuela tenía un patrimonio elevado, es posible que estuviera sujeta al Impuesto sobre el Patrimonio.
*   **Plazos:** Es fundamental cumplir con los plazos legales para presentar la declaración del Impuesto sobre Sucesiones y Donaciones.

**Pasos a seguir:**

1.  **Reúne la documentación:** Necesitarás el certificado de defunción de tu abuela, el testamento (si existe), el certificado del Registro de la Propiedad de la vivienda, y cualquier otro documento relevante.
2.  **Busca asesoramiento profesional:** Te recomiendo encarecidamente que consultes con un asesor fiscal o abogado especializado en herencias. Ellos podrán analizar tu situación particular en detalle y determinar con precisión los beneficios fiscales a los que tienes derecho, así como ayudarte con los trámites necesarios.

Es importante recordar que esta información es orientativa y no sustituye el asesoramiento profesional específico para tu caso.

## Estructura de archivos
Los archivos centrales del proyecto son agents.py y RAG.py. 
El primero contiene los prompts, los agentes y el grafo del sistema agéntico. El archivo RAG contiene las funciones para realizar los embeddings y todo el proceso descrito en el apartado 3, jugando la función "query_similar" un papel fundamental.
El archivo main.py sirve para levantar un servidor con FastAPI por donde consumir el sistema.

## Requisitos computacionales
Si bien los modelos Gemini se consumen vía API, el modelo de embeddings y el reranker se utilizan en local mediante uso acelarado por GPU.

## Modelos utilizados
- Familia de modelos Gemini 2.0 (thinking, flash, pro, lite) para cada agente, seleccionando el más adecuado según la tarea (pensamiento, generación rápida, redacción, etc.).
- Embeddings: https://huggingface.co/littlejohn-ai/bge-m3-spa-law-qa
- Reranker: https://huggingface.co/BAAI/bge-reranker-v2-m3
