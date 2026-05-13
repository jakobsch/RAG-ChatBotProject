from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TableItem
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path

# Pfad zum data/pdfs Ordner robust bestimmen, egal von wo das Skript läuft
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

def split_large_table(table_md: str, max_chars: int = 2000) -> list[str]:
    """
    Splittet eine zu lange Markdown-Tabelle in mehrere Teile.
    Header wird in jedem Teil mitgegeben. Wenn einzelne Datenzeilen
    bereits zu lang sind, werden sie hart nach Zeichenanzahl gesplittet.
    """
    lines = table_md.split("\n")

    if len(lines) < 3:
        return [table_md]

    header = "\n".join(lines[:2])
    data_lines = lines[2:]

    if len(table_md) <= max_chars:
        return [table_md]

    # Verfügbarer Platz pro Chunk nach Abzug des Headers
    available = max_chars - len(header) - 1

    # Wenn der Header schon zu groß ist, gibts kein sinnvolles Splitting
    if available < 200:
        return [table_md[i:i+max_chars] for i in range(0, len(table_md), max_chars)]

    chunks_out = []
    current_lines = []
    current_len = 0

    for line in data_lines:
        # Fall 1: Die Zeile selbst ist länger als der verfügbare Platz
        if len(line) > available:
            if current_lines:
                chunks_out.append(header + "\n" + "\n".join(current_lines))
                current_lines = []
                current_len = 0
            for i in range(0, len(line), available):
                piece = line[i:i+available]
                chunks_out.append(header + "\n" + piece)
            continue

        # Fall 2: Normale Zeile, passt sie noch dazu?
        if current_len + len(line) + 1 > available:
            chunks_out.append(header + "\n" + "\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += len(line) + 1

    if current_lines:
        chunks_out.append(header + "\n" + "\n".join(current_lines))

    return chunks_out


def load_and_chunk(pdf_path: str) -> list[Document]:
    """
    Liest PDF mit Docling, behandelt Tabellen separat, splittet Text
    header-aware in Chunks. Jeder Chunk hat Metadaten zu Quelle, Seite,
    Section und Typ (text/table).
    """

    # ---- Schritt 1: PDF einlesen mit Docling ----
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    filename = Path(pdf_path).name

    # ---- Schritt 2: Items durchlaufen, Tabellen abzweigen, Text sammeln ----
    table_chunks = []
    pages_text = {}
    pages_sections = {}
    current_section = "Unknown"

    for item, _level in result.document.iterate_items():
        # ---- Fall A: Tabelle gefunden ----
        if isinstance(item, TableItem):
            try:
                table_md = item.export_to_markdown()
            except Exception:
                table_md = str(item.text) if hasattr(item, 'text') else ""

            if not table_md.strip():
                continue

            page = item.prov[0].page_no if item.prov else 0

            # Tabelle ggf. in mehrere Teile splitten (für Embedding-Token-Limits)
            table_parts = split_large_table(table_md, max_chars=2000)

            for part_idx, part in enumerate(table_parts):
                table_chunks.append(Document(
                    page_content=part,
                    metadata={
                        "source": pdf_path,
                        "filename": filename,
                        "page": page,
                        "section": current_section,
                        "type": "table",
                        "table_part": part_idx,  # 0, 1, 2... wenn aufgeteilt
                        "table_parts_total": len(table_parts),  # 1 wenn nicht aufgeteilt
                    }
                ))
            continue

        # ---- Fall B: Item ohne Text ----
        if not hasattr(item, 'text') or not item.text:
            continue

        # ---- Fall C: Normaler Text-Item ----
        label_str = str(item.label).lower() if hasattr(item, 'label') else ""
        page = item.prov[0].page_no if item.prov else 0

        if 'title' in label_str:
            formatted = f"\n# {item.text}\n\n"
            current_section = item.text.strip()
        elif 'section_header' in label_str or 'header' in label_str:
            formatted = f"\n## {item.text}\n\n"
            current_section = item.text.strip()
        else:
            formatted = f"{item.text}\n\n"

        if page not in pages_text:
            pages_text[page] = ""
            pages_sections[page] = current_section
        pages_text[page] += formatted

    # ---- Schritt 3: Text-Chunks bauen (header-aware) ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n# ", "\n## ", "\n### ",
            "\n\n", "\n", ". ", " ", "",
        ],
    )

    text_chunks = []
    for page_num, page_text in pages_text.items():
        page_chunks = splitter.create_documents(
            texts=[page_text],
            metadatas=[{
                "source": pdf_path,
                "filename": filename,
                "page": page_num,
                "section": pages_sections[page_num],
                "type": "text",
            }]
        )
        text_chunks.extend(page_chunks)

    # ---- Schritt 4: Zusammenführen und Indices vergeben ----
    all_chunks = text_chunks + table_chunks

    total = len(all_chunks)
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total

    return all_chunks


if __name__ == '__main__':
    chunks = load_and_chunk(str(PDF_DIR / "TabellenTest.pdf"))

    # Statistik
    text_count = sum(1 for c in chunks if c.metadata.get("type") == "text")
    table_count = sum(1 for c in chunks if c.metadata.get("type") == "table")
    print(f"Anzahl Chunks: {len(chunks)} ({text_count} Text, {table_count} Tabellen)")

    # Beispiel: erster Text-Chunk
    print(f"\n--- Erster Text-Chunk ---")
    print(chunks[0].page_content[:300])
    print(f"Metadaten: {chunks[0].metadata}")

    # Beispiel: erster Tabellen-Chunk (falls vorhanden)
    table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
    if table_chunks:
        print(f"\n--- Erster Tabellen-Chunk ---")
        print(table_chunks[0].page_content[:300])
        print(f"Metadaten: {table_chunks[0].metadata}")