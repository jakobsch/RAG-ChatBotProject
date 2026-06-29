#!/usr/bin/.env python3
"""
eval_pypdf.py — misst die Zeit der neuen pypdf-Pipeline und schreibt alle
Chunks zur Ansicht in eine Datei.

ABLEGEN: neben ingestion.py (src/pdf_processing/).
AUSFÜHREN:
    python src/pdf_processing/eval_pypdf.py                 # erstes PDF in data/pdfs
    python src/pdf_processing/eval_pypdf.py data/pdfs/X.pdf  # bestimmtes PDF
"""

import sys
import time
from pathlib import Path

from ingestion import load_and_chunk, PDF_DIR


def main():
    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1])
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"Keine PDFs in {PDF_DIR}.")
            sys.exit(1)
        pdf = pdfs[0]

    print(f"PDF: {pdf.name}")

    # Zeit messen (3 Läufe, damit es nicht am Messrauschen liegt)
    times = []
    chunks = []
    for _ in range(3):
        t0 = time.perf_counter()
        chunks = load_and_chunk(str(pdf))
        times.append(time.perf_counter() - t0)
    median = sorted(times)[1]

    print(f"Zeit (Median aus 3): {median:.2f}s")
    print(f"Chunks: {len(chunks)}")

    # Alle Chunks in eine Datei schreiben
    out = [
        f"# Chunks (pypdf) — {pdf.name}",
        f"{len(chunks)} Chunks, {median:.2f}s",
        "",
        "---",
    ]
    for c in chunks:
        i = c.metadata.get("chunk_index", "?")
        page = c.metadata.get("page", "?")
        out.append(f"\n## Chunk {i} — Seite {page}\n")
        out.append("```")
        out.append(c.page_content)
        out.append("```")

    Path("chunks_pypdf.md").write_text("\n".join(out), encoding="utf-8")
    print("-> chunks_pypdf.md")
    print("Such die Tabellen-Stellen (Strg+F) und prüf, ob die Zahlen auffindbar sind.")


if __name__ == "__main__":
    main()
