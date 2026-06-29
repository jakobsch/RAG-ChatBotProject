#!/usr/bin/.env python3
"""
compare_tables.py — schreibt die Tabellen-Chunks einmal mit FAST und einmal mit
ACCURATE in zwei Markdown-Dateien, damit du die Tabellen-Qualität vergleichen kannst.

ABLEGEN: neben ingestion_docling_backup.py (src/pdf_processing/).
AUSFÜHREN:
    python src/pdf_processing/compare_tables.py                 # erstes PDF in data/pdfs
    python src/pdf_processing/compare_tables.py data/pdfs/X.pdf  # bestimmtes PDF

Ergebnis: tables_fast.md und tables_accurate.md im Ordner, aus dem du das Script
startest (also Projekt-Root). Öffne beide nebeneinander und leg das PDF daneben.

WICHTIG: Beide Läufe nutzen denselben chunk_document-Code, nur der Tabellen-Modus
unterscheidet sich. So vergleichst du wirklich FAST vs ACCURATE und nichts anderes.
Lösch die zwei .md-Dateien hinterher wieder, die müssen nicht ins Repo.
"""

import os
import sys
from pathlib import Path

# Euer Chunking + der PDF-Ordner-Pfad
from ingestion_docling_backup import chunk_document, PDF_DIR

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice


def make_converter(mode: TableFormerMode) -> DocumentConverter:
    """Converter mit explizit gesetztem Tabellen-Modus (FAST oder ACCURATE)."""
    po = PdfPipelineOptions()
    po.do_ocr = False
    po.do_table_structure = True
    po.table_structure_options.mode = mode
    po.accelerator_options = AcceleratorOptions(
        num_threads=os.cpu_count() or 4,
        device=AcceleratorDevice.AUTO,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}
    )


def dump_tables(pdf_path: Path, mode: TableFormerMode, outfile: str) -> int:
    """Parst das PDF im gegebenen Modus und schreibt alle Tabellen-Chunks als Markdown."""
    converter = make_converter(mode)
    result = converter.convert(str(pdf_path))
    chunks = chunk_document(result, str(pdf_path))

    tables = [c for c in chunks if c.metadata.get("type") == "table"]

    out = [
        f"# Tabellen-Chunks — Modus: {mode.value.upper()}",
        f"Datei: {pdf_path.name}",
        f"Anzahl Tabellen-Chunks: {len(tables)}",
        "",
        "---",
    ]
    for i, c in enumerate(tables, 1):
        page = c.metadata.get("page", "?")
        part = c.metadata.get("table_part", 0)
        total = c.metadata.get("table_parts_total", 1)
        # Wenn eine Tabelle aufgeteilt wurde, sieht man das hier
        suffix = f"  (Teil {part + 1} von {total})" if total > 1 else ""
        out.append(f"\n## Tabelle {i} — Seite {page}{suffix}\n")
        out.append(c.page_content)
        out.append("")

    Path(outfile).write_text("\n".join(out), encoding="utf-8")
    return len(tables)


def main():
    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1])
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"Keine PDFs in {PDF_DIR} gefunden.")
            sys.exit(1)
        pdf = pdfs[0]

    print(f"PDF: {pdf.name}\n")

    print("Parse mit FAST...")
    n_fast = dump_tables(pdf, TableFormerMode.FAST, "tables_fast.md")
    print(f"  {n_fast} Tabellen-Chunks  ->  tables_fast.md")

    print("Parse mit ACCURATE...")
    n_acc = dump_tables(pdf, TableFormerMode.ACCURATE, "tables_accurate.md")
    print(f"  {n_acc} Tabellen-Chunks  ->  tables_accurate.md")

    print("\nFertig. Öffne tables_fast.md und tables_accurate.md nebeneinander,")
    print("leg das PDF daneben und prüfe an 2-3 wichtigen Tabellen:")
    print("  - Stimmen die Zahlen (CO2, NOX usw.) exakt?")
    print("  - Sind Spalten und Zeilen richtig getrennt?")
    print("  - Fehlt eine Zeile oder ist was verrutscht?")
    print("Kein Unterschied bei den Zahlen -> FAST behalten. Sonst -> ACCURATE.")


if __name__ == "__main__":
    main()
