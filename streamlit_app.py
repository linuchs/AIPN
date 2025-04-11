import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configurazione della pagina
st.set_page_config(
    page_title="Classificatore di Testo",
    page_icon="🤖",
    layout="wide"
)

# Titolo e descrizione
st.title("Classificatore di Testo con AI 🤖")
st.markdown("""Questo strumento utilizza un modello di intelligenza artificiale 
per classificare il testo inserito. Inserisci il tuo testo e scopri come viene classificato!""")

def load_model(model_path):
    try:
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configura il device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Carica il modello con ottimizzazione della memoria
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map='auto',
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )
        model.eval()

        return tokenizer, model, device
    except Exception as e:
        raise Exception(f"Errore durante il caricamento del modello: {str(e)}")


def classify_text(text, tokenizer, model, device):
    # Prepara l'input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Esegui l'inferenza
    with torch.no_grad():
        outputs = model(**inputs)

    # Calcola le probabilità
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

def main():
    # Carica il modello
    try:
        with st.spinner("Caricamento del modello in corso..."):
            model_path = "llama_model"
            tokenizer, model, device = load_model(model_path)
        st.success("Modello caricato con successo!")
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}")
        return

    # Area di input del testo
    text_input = st.text_area(
        "Inserisci il testo da classificare:",
        height=150,
        placeholder="Scrivi qui il tuo testo..."
    )

    # Pulsante per la classificazione
    if st.button("Classifica", type="primary"):
        if text_input.strip():
            try:
                with st.spinner("Classificazione in corso..."):
                    predicted_class, confidence = classify_text(
                        text_input, tokenizer, model, device
                    )

                # Visualizza i risultati
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Classe Predetta",
                        value=f"Classe {predicted_class}"
                    )
                with col2:
                    st.metric(
                        label="Confidenza",
                        value=f"{confidence*100:.2f}%"
                    )

                # Barra di confidenza
                st.progress(confidence)

            except Exception as e:
                st.error(f"Errore durante la classificazione: {e}")
        else:
            st.warning("Per favore, inserisci del testo da classificare.")

    # Informazioni aggiuntive nella sidebar
    with st.sidebar:
        st.header("Informazioni")
        st.info("""
        Questo classificatore utilizza un modello di intelligenza artificiale 
        per analizzare e classificare il testo inserito. 
        
        Il modello restituisce:
        - La classe predetta
        - Il livello di confidenza della predizione
        """)

if __name__ == "__main__":
    main()