from typing import List, Optional
from pathlib import Path
import shutil
from typing import Callable
 
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from .embedding_config import EmbeddingConfig 

default_path = Path(__file__).parent.parent.parent / "data" / "chroma_vectorstore"

def _build_embeddings(config: EmbeddingConfig) -> Embeddings:
    if config.provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=config.model_name)

    if config.provider == "gwdg":
        return OpenAIEmbeddings(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            check_embedding_ctx_length=False,
        )

    raise ValueError(f"Unbekannter Embedding-Provider: {config.provider}")
 
 
class VectorStore:
    """
    VectorStore mit Hybrid Search (BM25 + Chroma) und Cross-Encoder Reranking.
 
    Pipeline:
        Query
          ├─ BM25Retriever       (Keyword-Suche, exakte Treffer)
          └─ ChromaRetriever     (Semantische Suche, Bedeutungsähnlichkeit)
                    │
               EnsembleRetriever (RRF-Fusion beider Ergebnislisten)
                    │
            CrossEncoderReranker (bewertet Query-Dokument-Paare neu)
                    │
                Top-K Dokumente
    """
 
    # Bewährte Standardmodelle – können im Konstruktor überschrieben werden

    DEFAULT_CROSS_ENCODER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
    def __init__(
            self,
            embedding_config: Optional[EmbeddingConfig] = None,
            persist_directory: Optional[str] = None,
            cross_encoder_model: str = DEFAULT_CROSS_ENCODER,
        ):
            self.embedding_config = embedding_config or EmbeddingConfig.from_env()
            self.persist_directory = persist_directory or str(
                default_path.parent / f"chroma_{self.embedding_config.cache_key}"
            )
            self._cross_encoder_model = cross_encoder_model

            self._embeddings: Optional[Embeddings] = None
            self._cross_encoder: Optional[HuggingFaceCrossEncoder] = None
            self._chroma_db: Optional[Chroma] = None
            self._bm25_docs: List[Document] = []
 
    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            print(f"Lade Embeddings ({self.embedding_config.cache_key})...", flush=True)
            self._embeddings = _build_embeddings(self.embedding_config)
        return self._embeddings

    @property
    def cross_encoder(self) -> HuggingFaceCrossEncoder:
        if self._cross_encoder is None:
            print("Lade Cross-Encoder...", flush=True)
            self._cross_encoder = HuggingFaceCrossEncoder(model_name=self._cross_encoder_model)
        return self._cross_encoder
    
    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------
 
    def _load_chroma(self) -> Chroma:
        """Lädt die persistente ChromaDB (lazy, wird nur einmal geladen)."""
        if self._chroma_db is None:
            self._chroma_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self._chroma_db
    
    def get_chunk_count(self) -> int:
        """Gibt die genaue Anzahl der Chunks in der persistenten ChromaDB zurück."""
        db = self._load_chroma()
        return db._collection.count()
 
    def _build_bm25_retriever(self, k: int) -> BM25Retriever:
        """Erstellt einen BM25Retriever aus den gespeicherten Dokumenten."""
        if not self._bm25_docs:
            raise ValueError(
                "Keine Dokumente für BM25 vorhanden. "
                "Rufe zuerst save_documents_to_db() auf."
            )
        retriever   = BM25Retriever.from_documents(self._bm25_docs)
        retriever.k = k
        return retriever
 
    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
 
    def _get_stored_sources(self) -> set[str]:
        """
        Gibt alle eindeutigen 'filename'-Werte (Dateinamen) zurück,
        die bereits in ChromaDB gespeichert sind.
        """
        db = self._load_chroma()
        result = db._collection.get(include=["metadatas"])
        sources = set()
        for metadata in result.get("metadatas", []):
            if metadata and "filename" in metadata:
                sources.add(Path(metadata["filename"]).name)
        return sources
    
    def is_document_stored(self, filename: str) -> bool:
        """Prüft, ob eine PDF (anhand des Dateinamens) bereits in der DB liegt."""
        return Path(filename).name in self._get_stored_sources()
    
    def clear_database(self) -> None:
        """
        Löscht die gesamte ChromaDB auf Dateisystemebene und setzt alle Caches zurück.
        Sicherer als collection.delete(), da keine verwaisten Daten übrig bleiben.
        """
        self._chroma_db = None
        self._bm25_docs = []

        path = Path(self.persist_directory)
        if path.exists():
            shutil.rmtree(path)
            print(f"ChromaDB unter '{path}' vollständig gelöscht.")
        else:
            print("Kein Datenbankverzeichnis gefunden – nichts zu löschen.")

    def save_documents_to_db(self, chunks: List[Document]) -> None:
        """
        Speichert Dokumente persistent in ChromaDB und hält sie
        zusätzlich im Speicher für BM25 vor.

        Bereits gespeicherte PDFs (erkannt am Dateinamen im Metadatenfeld 'filename')
        werden übersprungen — nur neue Chunks werden hinzugefügt (Upsert-Logik).
        """
        if not chunks:
            print("Keine Dokumente zum Speichern übergeben.")
            return

        # --- Deduplizierung: bereits gespeicherte Quellen ermitteln ---
        stored_sources = self._get_stored_sources()

        new_chunks = [
            chunk for chunk in chunks
            if Path(chunk.metadata.get("filename", "")).name not in stored_sources
        ]

        skipped_sources = {
            Path(chunk.metadata.get("filename", "")).name
            for chunk in chunks
        } - {Path(chunk.metadata.get("filename", "")).name for chunk in new_chunks}

        if skipped_sources:
            print(f"Bereits vorhanden (übersprungen): {', '.join(sorted(skipped_sources))}")

        if not new_chunks:
            print("Keine neuen Dokumente zum Speichern – alles bereits in der DB.")
            # BM25 trotzdem mit allen Chunks aktualisieren (für laufende Session)
            self._bm25_docs = chunks
            return

        # --- Neue Chunks in ChromaDB einfügen ---
        db = self._load_chroma()
        db.add_documents(new_chunks)

        # BM25-Kopie im Speicher mit allen Chunks aktualisieren
        self._bm25_docs = chunks

        print(f"{len(new_chunks)} neue Chunks aus {len({Path(c.metadata.get('filename','')).name for c in new_chunks})} PDF(s) gespeichert.")
 
    def load_documents_for_bm25(self, filename: str) -> None:
        """
        Lädt alle bereits in ChromaDB gespeicherten Chunks eines Dokuments
        (identifiziert über 'filename') in den BM25-Speicher, ohne ChromaDB
        zu verändern. Nützlich, wenn ein PDF bereits verarbeitet wurde und
        BM25 für die aktuelle Session (wieder-)aktiviert werden soll.
        """
        name = Path(filename).name
        db = self._load_chroma()

        result = db.get(where={"filename": name}, include=["documents", "metadatas"])

        chunks = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

        if not chunks:
            print(f"Keine gespeicherten Chunks für '{name}' gefunden.")
            return

        self._bm25_docs = chunks
        print(f"{len(chunks)} Chunks für '{name}' aus ChromaDB für BM25 geladen.")
 
    def as_hybrid_reranking_retriever(
        self,
        filename: str,
        k: int = 3,
        candidate_k: int = 20,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> ContextualCompressionRetriever:
        """
        Gibt einen Retriever zurück, der Hybrid Search und Reranking kombiniert.
 
        Ablauf:
            1. BM25 und Chroma rufen je `candidate_k` Dokumente ab.
            2. EnsembleRetriever fusioniert beide Listen via Reciprocal Rank Fusion.
            3. CrossEncoderReranker bewertet die fusionierten Kandidaten neu
               und gibt die besten `k` Dokumente zurück.
 
        Args:
            k:              Anzahl der finalen Dokumente nach dem Reranking.
            candidate_k:    Anzahl der Kandidaten pro Retriever vor dem Reranking.
                            Höher = besser Recall, aber langsamer.
            bm25_weight:    Gewicht des BM25-Retrievers in der Fusion (0–1).
            vector_weight:  Gewicht des Vektor-Retrievers in der Fusion (0–1).
                            bm25_weight + vector_weight sollte 1.0 ergeben.
 
        Returns:
            ContextualCompressionRetriever (LangChain Runnable, Chain-kompatibel)
        """
        if not self._bm25_docs:
            raise ValueError(
                "BM25-Dokumente fehlen. Rufe save_documents_to_db() oder "
                "load_documents_for_bm25() auf, bevor du diesen Retriever verwendest."
            )
 
        # --- Schritt 1: Basis-Retriever aufbauen ---
        bm25_retriever = self._build_bm25_retriever(k=candidate_k)
 
        vector_retriever = self._load_chroma().as_retriever(
            search_kwargs={"k": candidate_k, "filter": {"filename": Path(filename).name},}
        )

        # Instruction-Prefix nur für den Vektor-Zweig anwenden (z.B. E5-Mistral bei GWDG).
        # BM25 bekommt bewusst die unveränderte Query, da Keyword-Suche sonst verfälscht wird.
        if self.embedding_config.query_prefix:
            prefix = self.embedding_config.query_prefix
            vector_retriever = RunnableLambda(lambda q: prefix + q) | vector_retriever
 
        # --- Schritt 2: Hybrid-Fusion via Reciprocal Rank Fusion ---
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight],
        )
 
        # --- Schritt 3: Cross-Encoder Reranking ---
        reranker = CrossEncoderReranker(
            model=self.cross_encoder,
            top_n=k,
        )
 
        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble,
        )
 
 # Test Funktion

    def search_hybrid_reranked(
        self,
        query: str,
        k: int = 3,
        candidate_k: int = 20,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> List[Document]:
        """
        Direkte Suchmethode (ohne Retriever-Objekt) — praktisch für schnelle Tests.
 
        Gibt die `k` relevantesten Dokumente nach Hybrid Search + Reranking zurück.
        """
        if not query.strip():
            raise ValueError("Die Suchanfrage darf nicht leer sein.")
 
        retriever = self.as_hybrid_reranking_retriever(
            k=k,
            candidate_k=candidate_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
        return retriever.invoke(query)
    
if __name__ == "__main__":
    vc = VectorStore()
    vc._load_chroma()
    query = "Welche Ziele hat sich das Unternehmen in Bezug auf Nachhaltigkeit gesetzt?"
    retrieved_docs = vc.search_hybrid_reranked(query)
    print(len(retrieved_docs))