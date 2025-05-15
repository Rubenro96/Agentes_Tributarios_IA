import json
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Optional
import operator
from src.agent_handler import ConversationHistoryManager
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.RAG import VectorEmbeddings, DocumentType
import uuid
from pathlib import Path


# Cargamos las variables de entorno y configuramos la memoria para la persistencia
_ = load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class AgentState(TypedDict):
    pregunta: str
    tipo: str
    contenido: str
    consulta_RAG: str
    plan: str
    contexto: str
    respuesta: str
    revision: str
    num_revisiones: int
    max_revisiones: int
    thread_id: str
    conversation_history: Optional[List[Dict[str,str]]]

model_thinking = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", google_api_key=GOOGLE_API_KEY, temperature=0.3)
model_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, temperature=0.3)
model_lite = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", google_api_key=GOOGLE_API_KEY, temperature=0.3)
model_pro = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", google_api_key=GOOGLE_API_KEY, temperature=0.3)


rag = VectorEmbeddings(collection_name="normativa_tributaria-RAG")



PROMPT_TRIAGE = """Eres un agente de triaje, es decir, el primer eslabón de la cadena encargado de derivar el trabajo a agentes especializados conforme
el contenido de la cuestión planteada, para dar una mejor respuesta al usuario. 
En función del contenido de la pregunta del usuario, debes determinar entre dos escenarios:
1. Es una consulta específica para conocer información técnica sobre la normativa.
En el caso de que sea una consulta específica, debes elegir entre los siguientes temas y ámbitos que se ajusten a la consulta:
    GENERAL = "Normativa general aplicable (Ley General Tributaria, Ley 39/2015, Ley 40/2015, Ley de Financiación de las CC.AA.)"
    ITPAJD = "Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados" 
    ISD = "Impuesto sobre Sucesiones y Donaciones"
    IP = "Impuesto sobre el Patrimonio"
    JUEGO = "Tributos sobre el Juego"
    ANDALUCIA = "Comunidad Autónoma de Andalucía"
    ARAGON = "Comunidad Autónoma de Aragón"
    ASTURIAS = "Comunidad Autónoma del Principado de Asturias"
    BALEARES = "Comunidad Autónoma de las Islas Baleares"
    CANARIAS = "Comunidad Autónoma de Canarias"
    CANTABRIA = "Comunidad Autónoma de Cantabria"
    CASTILLA_LA_MANCHA = "Comunidad Autónoma de Castilla-La Mancha"
    CASTILLA_Y_LEON = "Comunidad Autónoma de Castilla y León"
    CATALUÑA = "Comunidad Autónoma de Cataluña"
    COMUNIDAD_VALENCIANA = "Comunidad Autónoma de la Comunidad Valenciana"
    EXTREMADURA = "Comunidad Autónoma de Extremadura"
    GALICIA = "Comunidad Autónoma de Galicia"
    LA_RIOJA = "Comunidad Autónoma de La Rioja"
    MURCIA = "Comunidad Autónoma de la Región de Murcia"
    MADRID = "Comunidad Autónoma de Madrid"
    NAVARRA = "Comunidad Foral de Navarra"
    PAIS_VASCO = "Comunidad Autónoma del País Vasco"

Pueden ser varias a la vez, por ejemplo, una consulta sobre los tipos impositivos reducidos en Cantabria en el ITPAJD podría contener: GENERAL (porque
da el marco general tributario), ITPAJD (es el tributo específico por el que nos están preguntando) y CANTABRIA (Comunidad Autónoma sobre la que va referida la pregunta).
La respuesta debe ser una lista de temas separados por ";". Por ejemplo: GENERAL; ITPAJD; CANTABRIA

2. Es una petición para reservar cita con un especialista de la agencia tributaria o para solicitar el aplazamiento o fraccionamiento de una deuda tributaria.
Puedes derivar la consulta al agente gestor.

En ambos casos emite la respuesta con la siguiente estructura json:
{
    "tipo": "consulta",
    "contenido": "GENERAL; ITPAJD; CANTABRIA"
}   

o 

{
    "tipo": "derivación",
    "contenido": "Agente gestor"
}

IMPORTANTE: Tu respuesta DEBE ser un objeto JSON válido que siga el esquema, no saques en formato markdown.
Si no encaja en ninguna de las categorías anteriores responde: "Lamento no poder ser de utilidad, no puedo contestar a esa cuestión"
    """

PROMPT_RAG_REFORMULADOR = """Eres un asistente especialista en tributos. Tienes mucha experiencia en traducir las necesidades de los contribuyentes en
lenguaje jurídico y técnico. Tu función consiste en, a partir de la pregunta realizada por el usuario, revisar la redacción para garantizar que 
el RAG pueda funcionar correctamente y proporcionar una respuesta adecuada a la consulta del usuario.

Debes redactar una consulta clara y precisa que permita al RAG proporcionar una respuesta adecuada. Para ello debes evitar que aparezcan 
siglas innecesarias o ambiguas, utilizar un lenguaje técnico-legal y darle sentido lógico. Por ejemplo:
En el caso del Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados (ITPAJD), ¿debe tributar la transmisión de un bien inmueble que se destina a vivienda habitual? ¿Qué tipos impositivos reducidos existen para el caso de Cantabria?

En lugar de vivienda, casa, piso, utiliza términos como "bien inmueble" para mayor precisión técnica. Para el caso de vehículos, coches o motos utiliza medios de transporte usados.
Para la residencia, transforma el municipio en su respectiva Comunidad Autónoma. Es importante que identifiques en la información aportada posibles entidades, rasgos del sujeto (edad, residencia, discapacidad, estado civil, ...) o de la operación (tipo de inmueble, coste de la operación, ...).

3. IMPORTANTE - CONSIDERA LAS PREGUNTAS PREVIAS: Cuando el usuario hace una pregunta nueva, debes analizar las preguntas anteriores para:
   - Identificar contexto implícito que pueda afectar la consulta actual
   - Recuperar información relevante de preguntas anteriores que no se mencione explícitamente en la nueva consulta
   - Consolidar la información de múltiples preguntas cuando forman parte de un mismo tema o caso
   - Resolver referencias implícitas como "ese impuesto", "la misma comunidad", "ese caso", etc.

No utilices la coletilla de "Basado en...", formula el texto directamente.
Devuelve tan solamente la respuesta reformulada.
"""


PROMPT_ESPECIALISTA = """Eres un asistente especialista en normativa tributaria de las agencias tributarias autonómicas de España. Debes tener en cuenta que vas a asistir a contribuyentes en preguntas sobres cuestiones referentes al
ámbito fiscal en España. Para ello se te va proveer de un contexto. Debes seguir los siguientes pasos:
1 - Analiza la consulta del contribuyente y comprende el contexto proporcionado.
2 - Si el contexto que se te ha proporcionado no es sufuciente o no está muy relacionado con la pregunta del usuario, debes responder: "Lo lamento mucho, no estoy autorizado para responder a esa pregunta."
3 - Si el contexto es suficiente y está relacionado con la pregunta, proporciona una respuesta clara, precisa y fundamentada en la normativa tributaria vigente en España.
4 - Revisa la respuesta para evitar dar una respuesta incoherente o incorrecta, puede ocasionar serios perjuicios al contribuyente.

Entre las competencias de las comunidades autonómas tan solo se encuentran aquellas que les han sido delegadas por el Estado en materia tributaria, como la gestión de ciertos impuestos cedidos y la capacidad normativa en determinados tributos.
Esos tributos son:
- Impuesto sobre Sucesiones y Donaciones.
- Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados.
- Tributos sobre el Juego.
- Impuesto sobre el Patrimonio.

Asimismo, podrás responder a cuestiones generales sobre normativa tributaria como plazos o procesos que aparezcan regulados en la Ley General Tributaria o las leyes 39/2015 y 40/2015.
Para ello, utiliza el siguiente contexto proporcionado:
{contexto}
"""


PROMPT_REDACTOR = """Eres un especialista de la redacción, vas a recibir un mensaje con contenido tributario técnico y la pregunta que ha ocasionado esa respuesta.
Conforme a esa pregunta, debes ajustar el texto de la respuesta para que responda a la pregunta que se ha formulado de manera más precisa.
Ten en cuenta que la respuesta procede de un RAG, elimina y evita hacer menciones al contexto que se la ha pasado, dando la impresión de que ha sido un mismo agente 
el que ha tramitado todo el proceso.
Debes utilizar un lenguaje riguroso, pero accesible para un contribuyente que no necesariamente tiene que tener conocimientos profundos técnicos tributarios.

Esta es la pregunta inicialmente formulada:
{pregunta}

Y esta es la respuesta del RAG:
{respuesta}
"""

# Función para crear string del contexto
def create_context_string(context):
    if len(context) > 0:
        contexto = ""
        for r in context:  
            contexto += f"Resultado {context.index(r) + 1}:\n"
            contexto += f"Documento de origen: {r['metadata']['título']}\n"
            contexto += f"Ámbito documento: {r['metadata']['ámbito']}\n"
            contexto += f"Fecha de carga del documento: {r['metadata']['processing_date']}\n"
            contexto += f"\nContenido:\n{r['document']}\n\n" 
        return contexto
    else:
        contexto = "No dispones de información relevante para poder responder la consulta."
        return contexto

# Instanciamos el administrador de historial
conversation_manager = ConversationHistoryManager()

def triage_agent(state:AgentState):
    if state.get("thread_id"):
        history = conversation_manager.get_conversation_history(state["thread_id"])
        state["conversation_history"] = history
    
    system_prompt = PROMPT_TRIAGE
    if state.get("conversation_history") and len(state["conversation_history"]) > 0:
        history_str = "Historial de conversación previo:\n"
        for interaction in state["conversation_history"]:
            history_str += f"Usuario: {interaction['pregunta']}\n"
            history_str += f"Asistente: {interaction['revision']}\n\n"
        
        system_prompt += f"\n\nTen en cuenta el siguiente historial de conversación con el usuario:\n{history_str}"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["pregunta"])]
    response = model_flash.invoke(messages)
    try:
        contenido_json = response.content
        contenido_json = contenido_json.replace("```", "").replace("json\n", "")
        json_response = json.loads(contenido_json)
        tipo = json_response["tipo"]
        contenido = json_response["contenido"]
        return {"tipo": tipo, "contenido": contenido}
    except json.JSONDecodeError:
        return {"tipo": "error", "contenido": "Error al procesar la respuesta"}
    


def reformulador(state: AgentState):
    system_prompt = PROMPT_RAG_REFORMULADOR

    previous_questions = []
    current_question = state["pregunta"]

    if state.get("conversation_history"):
        for interaction in state["conversation_history"]:
            previous_questions.append(interaction['pregunta'])
        
        if previous_questions:
            questions_history = "HISTORIAL DE PREGUNTAS PREVIAS:\n"
            for i, question in enumerate(previous_questions):
                questions_history += f"Pregunta {i+1}: {question}\n"
            
            questions_context = f"""
            {questions_history}

            INSTRUCCIONES PARA USAR EL HISTORIAL:
            - La pregunta actual es: "{current_question}"
            - Analiza las preguntas anteriores para identificar información relevante que pueda complementar la consulta actual.
            - Si la pregunta actual parece ser una continuación o hace referencia a preguntas previas, incorpora ese contexto.
            - Si hay términos ambiguos o referencias implícitas (como "eso", "ese impuesto", etc.), resuélvelos basándote el historial.
            - Refomula la consulta actual para que sea completa por sí misma, pero incorporando el contexto relevante de preguntas previas.
            """

            system_prompt += f"\n\n {questions_context}"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=current_question)]
    response = model_flash.invoke(messages)
    return {"consulta_RAG": response.content}
    

def specialist_node(state:AgentState):
    tipos_docs = []
    for d in state["contenido"].split(";"):
        tipo_doc = getattr(DocumentType, d.strip().upper())
        tipos_docs.append(tipo_doc)

    try:
        result = rag.query_similar(query_text=state["consulta_RAG"], score=0.6, document_types=tipos_docs)
        contexto = create_context_string(result)
        messages = [
            SystemMessage(content=PROMPT_ESPECIALISTA.format(contexto=contexto)), 
            HumanMessage(content=state["pregunta"])
        ]
        response = model_lite.invoke(messages)
        return {
            "respuesta": response.content,
            "contexto": contexto
        }
    except Exception as e:
        # Manejo de errores
        print(f"Error en specialist_node: {str(e)}")
        return {
            "respuesta": "Lo siento, ha ocurrido un error al procesar su consulta.",
            "contexto": "No se pudo obtener el contexto debido a un error."
        }

def need_specialist(state: AgentState):
    if state.get("tipo") == "consulta":
        return "reformulador"
    elif state.get("tipo") == "derivación":
        return END
    else:
        return END


def redactor(state: AgentState):
    pregunta = state["pregunta"]
    respuesta = state["respuesta"]
    prompt_final = PROMPT_REDACTOR.format(pregunta=pregunta, respuesta=respuesta)
    response = model_flash.invoke(prompt_final)
    
    return {"revision": response.content}
    

# Configuración del grafo
builder = StateGraph(AgentState)

# Agregar nodos
builder.add_node("triage_agent", triage_agent)
builder.add_node("reformulador", reformulador)
builder.add_node("especialista", specialist_node)
builder.add_node("redactor", redactor)

# Establecer el punto de entrada
builder.set_entry_point("triage_agent")

# Agregar arista condicional desde triage_agent
builder.add_conditional_edges("triage_agent",need_specialist)

# Agregar aristas no condicionales
builder.add_edge("reformulador", "especialista")
builder.add_edge("especialista", "redactor")
builder.add_edge("redactor", END)

# Compilar el grafo
graph = builder.compile()


def run_conversation(question, thread_id=None):
    if not thread_id:
        thread_id = f"thread_{uuid.uuid4()}"
        print(f"Se ha creado nuevo hilo de conversación: {thread_id}")
    else:
        print(f"Continuando la conversación en el hilo: {thread_id}")

    initial_state = {
        "pregunta": question,
        "thread_id": thread_id,
        "num_revisiones": 0,
        "max_revisiones": 2
    }
    result = graph.invoke(initial_state)

    conversation_manager.log_interaction(
        thread_id = thread_id,
        pregunta = question,
        tipo = result["tipo"],
        contenido=result["contenido"],
        consulta_RAG=result["consulta_RAG"],
        plan="",
        contexto=result["contexto"],
        respuesta=result["respuesta"],
        revision=result["revision"]
    )
    return result
    



