"""
cut_pdf.py — schneidet einen Seitenbereich aus einem PDF und speichert ihn neu.
Behält hier Seite 66-76, verwirft den Rest. Lokal, kein Upload, keine Limits.

AUSFÜHREN (im venv, aus dem Projekt-Root):
    python cut_pdf.py
"""

from pypdf import PdfReader, PdfWriter

# ===================== HIER ANPASSEN =====================
INPUT = "data/pdfs/ZahlenTabellen.pdf"   # <- Dateiname des großen PDFs eintragen
OUTPUT = "data/pdfs/ZahlenTabellengekürzt.pdf"        # <- so heißt das Ergebnis
START_SEITE = 66    # erste Seite, die behalten wird (wie im PDF-Viewer oben gezählt)
END_SEITE = 76      # letzte Seite, inklusive
# =========================================================

reader = PdfReader(INPUT)
total = len(reader.pages)
print(f"Eingabe: {INPUT} ({total} Seiten)")

# Sicherheits-Check, falls der Bereich über das PDF hinausgeht
if END_SEITE > total:
    print(f"ACHTUNG: PDF hat nur {total} Seiten - kürze Ende auf {total}.")
    END_SEITE = total

writer = PdfWriter()
# START_SEITE-1, weil pypdf intern ab 0 zählt; Slice bis END_SEITE = inklusive END_SEITE
for page in reader.pages[START_SEITE - 1:END_SEITE]:
    writer.add_page(page)

with open(OUTPUT, "wb") as f:
    writer.write(f)

print(f"Fertig: {len(writer.pages)} Seiten gespeichert -> {OUTPUT}")
