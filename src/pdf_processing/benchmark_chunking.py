#!/usr/bin/.env python3
"""
benchmark_chunking.py — misst, wo die Zeit in der Ingestion-Pipeline draufgeht:
Parsing (Docling) vs. Chunking (LangChain Splitter).

ABLEGEN: in denselben Ordner wie ingestion_docling_backup.py.
AUSFÜHREN (im Projekt-venv, da wo docling + langchain installiert sind):

    python benchmark_chunking.py                  # nutzt data/pdfs
    python benchmark_chunking.py /pfad/zu/pdfs     # eigener Ordner
    python benchmark_chunking.py --runs 3          # 3 Durchläufe pro PDF, Median
    python benchmark_chunking.py --compare-tables  # misst zusätzlich die Kosten
                                                   # der Tabellen-Erkennung

Der Benchmark ruft GENAU euren Code auf (parse_pdf + chunk_document aus
ingestion_docling_backup.py), misst also die echten Zahlen, keinen Nachbau.
"""

import sys
import time
import argparse
import statistics
from pathlib import Path

# Euer echter Code - die zwei Phasen separat
from ingestion_docling_backup import build_converter, parse_pdf, chunk_document, PDF_DIR

# Nur für den optionalen Tabellen-Vergleich gebraucht
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def detect_device() -> str:
    """Sagt, ob Docling auf GPU oder CPU läuft - oft der größte versteckte Hebel."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return "GPU (Apple MPS)"
    except Exception:
        pass
    return "CPU"


def page_count(doc) -> int:
    """Seitenzahl robust auslesen (Docling-API variiert je nach Version)."""
    try:
        return len(doc.pages)
    except Exception:
        try:
            return doc.num_pages()
        except Exception:
            return 0


def build_converter_no_tables() -> DocumentConverter:
    """Wie euer Converter, aber Tabellen-Erkennung AUS - um deren Kosten zu isolieren."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def time_pdf(pdf_path: Path, converter) -> dict:
    """Misst Parse- und Chunk-Zeit für ein einzelnes PDF."""
    # --- Phase 1: Parsing ---
    t0 = time.perf_counter()
    result = parse_pdf(str(pdf_path), converter=converter)
    t_parse = time.perf_counter() - t0

    pages = page_count(result.document)

    # --- Phase 2: Chunking (euer chunk_document) ---
    t1 = time.perf_counter()
    chunks = chunk_document(result, str(pdf_path))
    t_chunk = time.perf_counter() - t1

    n_tables = sum(1 for c in chunks if c.metadata.get("type") == "table")
    return {
        "file": pdf_path.name,
        "pages": pages,
        "chunks": len(chunks),
        "tables": n_tables,
        "t_parse": t_parse,
        "t_chunk": t_chunk,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=None,
                    help="Ordner mit PDFs (default: data/pdfs aus ingestion_docling_backup.py)")
    ap.add_argument("--runs", type=int, default=1,
                    help="Wiederholungen pro PDF, Median wird genommen (gegen Messrauschen)")
    ap.add_argument("--compare-tables", action="store_true",
                    help="Zusätzlich ohne Tabellen-Erkennung messen, um deren Kosten zu zeigen")
    args = ap.parse_args()

    folder = Path(args.folder) if args.folder else PDF_DIR
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"Keine PDFs in {folder} gefunden.")
        sys.exit(1)

    print(f"Device: {detect_device()}")
    print(f"PDFs:   {len(pdfs)} in {folder}")
    print(f"Runs:   {args.runs} (Median)\n")

    # Converter EINMAL bauen und wiederverwenden (so misst man Parsing fair,
    # ohne Converter-Aufbau jedes Mal mitzuzählen).
    converter = build_converter()

    # Warm-up: der erste convert lädt die Docling-Modelle herunter/in den Speicher.
    # Beim allerersten Mal evtl. ein paar Minuten - nicht abbrechen, nicht mitmessen.
    print("Warm-up (Modelle laden)...")
    _ = parse_pdf(str(pdfs[0]), converter=converter)
    print("fertig.\n")

    rows = []
    for pdf in pdfs:
        runs = [time_pdf(pdf, converter) for _ in range(args.runs)]
        r = dict(runs[0])
        r["t_parse"] = statistics.median(x["t_parse"] for x in runs)
        r["t_chunk"] = statistics.median(x["t_chunk"] for x in runs)
        rows.append(r)
        print(f"  {pdf.name:<35.35} {r['pages']:>4}p  "
              f"parse {r['t_parse']:>7.2f}s  chunk {r['t_chunk']:>7.3f}s  "
              f"({r['chunks']:>4} chunks, {r['tables']} tab)")

    # ---- Aggregat ----
    tot_parse = sum(r["t_parse"] for r in rows)
    tot_chunk = sum(r["t_chunk"] for r in rows)
    tot_pages = sum(r["pages"] for r in rows)
    tot_chunks = sum(r["chunks"] for r in rows)
    total = tot_parse + tot_chunk or 1e-9

    print("\n" + "=" * 64)
    print(f"GESAMT: {len(rows)} PDFs, {tot_pages} Seiten, {tot_chunks} Chunks")
    print("-" * 64)
    print(f"  Parsing (Docling)    {tot_parse:>9.2f}s   {tot_parse/total*100:>5.1f}%")
    print(f"  Chunking (Splitter)  {tot_chunk:>9.2f}s   {tot_chunk/total*100:>5.1f}%")
    print("-" * 64)
    if tot_pages:
        print(f"  Parsing pro Seite:      {tot_parse/tot_pages*1000:>7.0f} ms")
    if tot_chunk > 0:
        print(f"  Parse/Chunk-Verhältnis: {tot_parse/tot_chunk:>7.0f}x")
    print("=" * 64)

    # ---- Verdict ----
    if tot_parse / total > 0.8:
        print("\n→ Flaschenhals ist eindeutig das PARSING (Docling), nicht das Chunking.")
        print("  Am Splitter zu optimieren bringt nichts.")
        print("  Echte Hebel: über Dokumente parallelisieren, Parse-Ergebnis cachen, GPU.")
    elif tot_chunk / total > 0.3:
        print("\n→ Überraschung: Chunking hat einen relevanten Anteil. Genauer hinschauen lohnt.")

    # ---- Optional: Kosten der Tabellen-Erkennung isolieren ----
    if args.compare_tables:
        print("\n" + "-" * 64)
        print("Vergleich: Parsing MIT vs. OHNE Tabellen-Erkennung")
        print("(Achtung: parst nochmal alles, dauert also extra)")
        conv_notab = build_converter_no_tables()
        _ = conv_notab.convert(str(pdfs[0]))  # warm-up für den zweiten Converter

        tot_with, tot_without = 0.0, 0.0
        for pdf in pdfs:
            t0 = time.perf_counter()
            parse_pdf(str(pdf), converter=converter)
            tot_with += time.perf_counter() - t0

            t0 = time.perf_counter()
            conv_notab.convert(str(pdf))
            tot_without += time.perf_counter() - t0

        saved = tot_with - tot_without
        print(f"  mit Tabellen:  {tot_with:>8.2f}s")
        print(f"  ohne Tabellen: {tot_without:>8.2f}s")
        if tot_with > 0:
            print(f"  → Tabellen-Erkennung kostet {saved:.2f}s "
                  f"({saved/tot_with*100:.0f}% der Parse-Zeit)")
        print("    Wenn das viel ist: TableFormerMode.FAST testen - aber Tabellen-")
        print("    Qualität gegenchecken, ihr verlasst euch ja darauf.")


if __name__ == "__main__":
    main()
