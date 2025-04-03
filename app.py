"""Applicazione Streamlit per chattare con Llama3.2 utilizzando RAG con parametri modificabili."""

import os
import requests
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# App title
st.title("Chat con Llama3.2 🦙 + RAG (Parametri Modificabili)")

# Sidebar settings
st.sidebar.header("Impostazioni del Modello")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1
)
top_p = st.sidebar.slider(
    "Top-p",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.1
)

# Inizializza lo storico della conversazione
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def query_ollama(prompt):
    """Invia una richiesta al modello LLM Ollama.

    Args:
        prompt (str): Il prompt di input per il modello.

    Returns:
        str: La risposta del modello o il messaggio di errore.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "Errore nella risposta del modello.")
    except requests.exceptions.RequestException as e:
        return f"Errore durante la comunicazione con Ollama: {e}"


def load_and_prepare_pdf(file_path):
    """Carica e prepara il documento PDF per RAG.

    Args:
        file_path (str): Percorso del file PDF.

    Returns:
        FAISS: Vector store contenente gli embedding del documento.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store


def retrieve_relevant_info(vector_store, query, k=3):
    """Recupera informazioni rilevanti dal vector store.

    Args:
        vector_store (FAISS): Il vector store contenente gli embedding del documento.
        query (str): La query di ricerca.
        k (int, optional): Numero di documenti da recuperare. Default a 3.

    Returns:
        str: Contenuti dei documenti rilevanti concatenati.
    """
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context


# PDF upload section
st.subheader("Carica un Documento PDF")
uploaded_file = st.file_uploader("Seleziona un file PDF", type=["pdf"])

vector_store = None
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    vector_store = load_and_prepare_pdf("temp.pdf")
    #os.remove("temp.pdf")  

# Interfaccia utente
st.subheader("Conversazione")

# Visualizza lo storico della conversazione
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
user_input = st.chat_input("Inserisci il tuo messaggio...")

if user_input and vector_store:
    # Aggiungi il messaggio dell'utente allo storico
    st.session_state.conversation_history.append(
        {"role": "user", "content": user_input}
    )

    # Visualizza il messaggio dell'utente
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Recupero informazioni e generazione della risposta..."):
        # Recupera informazioni rilevanti dal vector store
        context = retrieve_relevant_info(vector_store, user_input)

        # Crea il prompt completo per il modello
        full_prompt = (
            f"Domanda: {user_input}\n\n"
            f"Contesto dal documento:\n{context}\n\n"
            "Rispondi alla domanda basandoti sul contesto fornito."
        )

        # Invia la richiesta al modello
        response = query_ollama(full_prompt)

    # Aggiungi la risposta del modello allo storico
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": response}
    )

    # Visualizza la risposta del modello
    with st.chat_message("assistant"):
        st.markdown(response)

elif user_input and not vector_store:
    st.warning("Per favore, carica prima un documento PDF.")
