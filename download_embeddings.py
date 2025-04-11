from sentence_transformers import SentenceTransformer
import os

def download_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2", save_path="embeddings_model"):
    """
    Scarica e salva localmente un modello di embedding.
    
    Args:
        model_name (str): Nome del modello da scaricare da Hugging Face
        save_path (str): Percorso dove salvare il modello
    """
    try:
        # Crea la directory se non esiste
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Downloading model {model_name}...")
        # Scarica e salva il modello
        model = SentenceTransformer(model_name)
        model.save(save_path)
        print(f"Model successfully downloaded and saved to {save_path}")
        
        # Verifica il funzionamento
        test_text = "This is a test sentence."
        embeddings = model.encode(test_text)
        print(f"Test successful! Generated embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"Error during model download: {e}")

if __name__ == "__main__":
    download_embedding_model()