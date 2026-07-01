"""


wechsel docling auf pypdf
->Docling war zu langsam->durch ML Modelle

Tradeoff: pypdf erkennt keine Tabellenstruktur.

"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Pfad zum data/pdfs Ordner robust bestimmen, egal von wo Skript läuft
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"


def load_and_chunk(pdf_path: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[Document]:
    """
    Liest ein PDF mit pypdf und zerlegt es in Chunks.

    pdf_path: Pfad zur PDF-Datei
    chunk_size / chunk_overlap: 1000/200
    Rückgabe: Liste von Document-Objekten mit Metadaten (gleiche Keys wie vorher).
    """
    filename = Path(pdf_path).name

    #Schritt 1: PDF laden
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Seitenzahl in den Text voranstellen, damit sie nach dem Splitten erhalten
    # bleibt (so kann das LLM später die Quelle nennen).
    for p in pages:
        page_no = p.metadata.get("page", "?")
        p.page_content = f"(Seite {page_no})\n{p.page_content}"

    # Schritt 2: chunk zerlegung
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    #Schritt 3: Metadaten vereinheitlichen
    #Saubere, einfach-typisierte Metadaten
    total = len(chunks)
    for i, c in enumerate(chunks):
        page_no = c.metadata.get("page", "?")
        page_no = page_no + 1 if isinstance(page_no, int) else page_no
        c.metadata = {
            "source": pdf_path,
            "filename": filename,
            "page": page_no,
            "section": "",
            "type": "text",         
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
