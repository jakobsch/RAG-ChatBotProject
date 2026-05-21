from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStorageModule():

    def __init__(self, persist_directory: str = "./data/chroma_vectorstore"):
        
        # Definition des verwendeten Embedding Model

        self.embeddings = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")

        # Getrennte Collection für reine Text Daten und Tabellen Daten zur RAG Optimierung

        self.text_store = Chroma(
            collection_name="pdf_texts",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        self.table_store = Chroma(
            collection_name="pdf_tables",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def check_for_document_in_db():
        pass

    def store_chunks (self, chunks: List[Document]):

        if not chunks:
            print ("No Chunks for storage detected")

        text_chunks = [c for c in chunks if c.metadata.get("type") == "text"]
        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]

        try:

            if text_chunks:
                self.text_store.add_documents(text_chunks)
                print(f"{len(text_chunks)} Chunks successfully stored")

        except Exception as e:
            print(f"The following exception occured while storing the text chunks: {e}")

        try:

            if table_chunks:
                self.table_store.add_documents(table_chunks)
                print(f"{len(table_chunks)} Chunks successfully stored")

        except Exception as e:

            print(f"The following exception occured while storing the table chunks: {e}")

    def query_store():
        pass


