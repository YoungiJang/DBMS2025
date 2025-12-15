from langchain_huggingface.embeddings import HuggingFaceEmbeddings

MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    print(f"Loading embedding model: {MODEL_NAME}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={'normalize_embeddings':True}
    )
    print("--- Embedding model ready ---")
    return embeddings