from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

default_path = Path("data/chroma_vectorstore")

class VectorStore ():

    def __init__(self, persist_directory: str = str(default_path), model_name: str = "all-MiniLM-L6-v2"):
    
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model= model_name)

    def save_documents_to_db(self, chunks: List[Document]) -> None:
        """Nimmt eine Liste von Langchain-Dokumenten entgegen und speichert sie persistent."""
        if not chunks:
            print("Keine Dokumente zum Speichern übergeben.")
            return

        # Initialisiert Chroma mit dem Speicherpfad und fügt Dokumente hinzu
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"{len(chunks)} Chunks erfolgreich in {self.persist_directory} gespeichert.")


    def retrieve_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """Sucht nach den ähnlichsten Dokumenten basierend auf einer User-Query."""
        if not query.strip():
            raise ValueError("Die Suchanfrage darf nicht leer sein.")

        # Lädt die existierende persistente Datenbank
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        # Führt die Ähnlichkeitssuche aus
        similar_docs = db.similarity_search(query, k=k)
        return similar_docs