"""
ingestion.py — PDF laden + chunken mit pypdf.

WECHSEL VON DOCLING AUF PYPDF:
Docling war auf CPU ~40s/Dokument, weil es ML-Modelle für Layout und Tabellen
fährt. pypdf liest nur den rohen Text -> praktisch sofort. Das löst die
Geschwindigkeits-Kritik direkt.

TRADE-OFF: pypdf erkennt KEINE Tabellenstruktur. Tabellen-Zahlen kommen als
Fließtext, nicht als Zeile/Spalte. Für zusammenfassende Fragen ("was sagt der
Bericht über CO2?") reicht das - so hat es die 1,x-Gruppe bei identischer
Aufgabe gemacht.

WICHTIG: load_and_chunk() behält Signatur und Metadaten-Schema bei, damit
app.py, der Vectorstore und rag_chain.py unverändert weiterlaufen.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Pfad zum data/pdfs Ordner robust bestimmen, egal von wo das Skript läuft
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"


def load_and_chunk(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Liest ein PDF mit pypdf und zerlegt es in Chunks.

    pdf_path: Pfad zur PDF-Datei
    chunk_size / chunk_overlap: 1000/200 wie im Projekt-Standard (..env)
    Rückgabe: Liste von Document-Objekten mit Metadaten (gleiche Keys wie vorher).
    """
    filename = Path(pdf_path).name

    # ---- Schritt 1: PDF laden (pypdf, eine Document pro Seite, kein ML-Modell) ----
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Seitenzahl in den Text voranstellen, damit sie nach dem Splitten erhalten
    # bleibt (so kann das LLM später die Quelle nennen).
    for p in pages:
        page_no = p.metadata.get("page", "?")
        p.page_content = f"(Seite {page_no})\n{p.page_content}"

    # ---- Schritt 2: In Chunks zerlegen ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # ---- Schritt 3: Metadaten vereinheitlichen ----
    # Saubere, einfach-typisierte Metadaten (Chroma verträgt keine komplexen Typen)
    # und dieselben Keys wie die alte Docling-Version, damit UI + Retrieval passen.
    total = len(chunks)
    for i, c in enumerate(chunks):
        page_no = c.metadata.get("page", "?")
        c.metadata = {
            "source": pdf_path,
            "filename": filename,
            "page": page_no,
            "section": "",          # pypdf liefert keine Section-Überschriften
            "type": "text",         # kein separater Tabellen-Typ mehr
            "chunk_index": i,
            "total_chunks": total,
        }

    return chunks


if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Keine PDFs in {PDF_DIR}.")
    else:
        chunks = load_and_chunk(str(pdfs[0]))
        print(f"Datei: {pdfs[0].name}")
        print(f"Anzahl Chunks: {len(chunks)}")
        print("\n--- Erster Chunk ---")
        print(chunks[0].page_content[:300])
        print(f"\nMetadaten: {chunks[0].metadata}")
