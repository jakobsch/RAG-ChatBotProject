from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


persistent_directory = "/home/linus/projekte/RAG-ChatBotProject/data/chroma_vectorstore"

embeddings = HuggingFaceEmbeddings("all-Mini LM-V6-v2")

def save_documents_to_db(chunks: List[Document]) -> None:
    """Nimmt eine Liste von Langchain-Dokumenten entgegen und speichert sie persistent."""
    if not chunks:
        print("Keine Dokumente zum Speichern übergeben.")
        return

    # Initialisiert Chroma mit dem Speicherpfad und fügt Dokumente hinzu
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persistent_directory=persistent_directory,
    )
    print(f"{len(chunks)} Chunks erfolgreich in {persistent_directory} gespeichert.")


def retrieve_similar_documents(query: str, k: int = 3) -> List[Document]:
    """Sucht nach den ähnlichsten Dokumenten basierend auf einer User-Query."""
    if not query.strip():
        raise ValueError("Die Suchanfrage darf nicht leer sein.")

    # Lädt die existierende persistente Datenbank
    db = Chroma(
        persistent_directory=persistent_directory,
        embedding_function=embeddings,
    )

    # Führt die Ähnlichkeitssuche aus
    similar_docs = db.similarity_search(query, k=k)
    return similar_docs