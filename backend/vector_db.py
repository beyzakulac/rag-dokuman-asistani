from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from trace_logger import trace_step

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@trace_step(step_name="vektor_veritabani_olusturma")
def create_vector_db(chunks):
    """
    Oluşturulan metin parçalarını (chunks) alıp geçici bellek (RAM)
    üzerinde çalışan FAISS vektör veritabanına kaydeder.
    """
    if not chunks:
        raise ValueError("Vektör veritabanı oluşturmak için metin parçası (chunk) bulunamadı.")

    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db


@trace_step(step_name="normal_arama")
def search_normal(query: str, vector_db: FAISS, k: int = 4):
    """Normal benzerlik araması (Similarity Search)."""
    return vector_db.similarity_search(query, k=k)


@trace_step(step_name="mmr_arama")
def search_mmr(query: str, vector_db: FAISS, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5):
    """Maximal Marginal Relevance (MMR) araması."""
    return vector_db.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )