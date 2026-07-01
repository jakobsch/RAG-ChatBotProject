"""
src/rag/rag_chain.py

Baut die RAG-Chain: Frage + relevante Chunks -> Antwort vom LLM.

Aktuell: FAKE_MODE = True  -> kein echter API-Call, Platzhalter-Antwort
Sobald API Key vorhanden: FAKE_MODE = False setzen und GWDG-Zugangsdaten in ..env eintragen
"""

import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional
from langchain_core.documents import Document

# ============================================================
# FAKE MODE SCHALTER
# True  = kein LLM nötig, gibt Platzhalter-Antwort zurück
# False = echter GWDG API Call (braucht API Key in ..env)
# ============================================================
FAKE_MODE = False


def build_prompt(question: str, chunks: list[Document]) -> str:
    """
    Baut einen ausführlichen RAG-Prompt aus User-Frage und gefundenen Chunks.

    Ziel:
    - Das LLM soll nur auf Basis der hochgeladenen PDF antworten.
    - Es soll keine Zahlen, Ziele oder Fakten erfinden.
    - Es soll Nachhaltigkeitsberichte besonders gut auswerten.
    - Es soll Tabellen, Kennzahlen, Jahre, Einheiten und Quellenhinweise beachten.
    """

    context_parts = []

    for i, doc in enumerate(chunks, start=1):
        metadata = doc.metadata or {}

        page = metadata.get("page", "?")
        section = metadata.get("section", "")
        chunk_type = metadata.get("type", "text")
        filename = metadata.get("filename", "")

        context_parts.append(
            f"[Quelle {i}]\n"
            f"Datei: {filename}\n"
            f"Seite: {page}\n"
            f"Abschnitt: {section}\n"
            f"Inhaltstyp: {chunk_type}\n\n"
            f"Inhalt:\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""

Nutze dafür ausschließlich den Kontext weiter unten.

==================================================
NUTZERFRAGE
==================================================

{question}

WICHTIG – SPRACHE: Antworte in der Sprache dieser Frage, unabhängig von der Sprache
der folgenden Regeln, Beispiele oder des Kontexts. Diese Regel gilt für den gesamten
Output inklusive Struktur-Labels und Quellenangaben.

==================================================
ANTWORTREGELN
==================================================

1. Kontexttreue
- Verwende nur Informationen, die im Kontext stehen.
- Wenn die Antwort nicht aus dem Kontext hervorgeht, schreibe:
  "Diese Information konnte ich im bereitgestellten Bericht nicht finden."
- Wenn nur teilweise Informationen vorhanden sind, sage klar, was gefunden wurde und was fehlt.
- Erfinde keine Werte, keine Prozentzahlen, keine Jahreszahlen, keine Zieljahre und keine Einheiten.

2. Umgang mit Zahlen und Kennzahlen
- Übernimm Zahlen exakt so, wie sie im Kontext stehen.
- Achte besonders auf Einheiten wie tCO2e, CO2, NOX, kWh, MWh, %, Fahrzeuge, Tonnen oder Jahre.
- Wenn mehrere Jahre genannt werden, vergleiche sie nur, wenn beide Werte im Kontext stehen.
- Wenn die Frage eine Entwicklung verlangt, zum Beispiel "Wie hat sich CO2 entwickelt?", dann nenne:
  - den Ausgangswert,
  - den späteren Wert,
  - die Richtung der Entwicklung,
  - und nur dann eine Differenz oder Prozentänderung, wenn sie im Kontext steht oder direkt eindeutig berechenbar ist.
- Wenn Werte aus verschiedenen Tabellen oder Abschnitten stammen, weise darauf hin.

3. Umgang mit Tabellen
- Nachhaltigkeitsberichte enthalten oft wichtige Informationen in Tabellen.
- Wenn der Kontext Markdown-Tabellen enthält, interpretiere sie sorgfältig.
- Achte auf Spaltenüberschriften, Zeilennamen, Jahre und Einheiten.
- Wenn eine Tabelle unvollständig oder abgeschnitten wirkt, formuliere vorsichtig.
- Verwechsle keine Spalten, Jahre oder Einheiten.
- Wenn ein Wert nicht eindeutig einer Spalte zugeordnet werden kann, sage das.

4. Sprache
- Antworte in derselben Sprache wie die Nutzerfrage.
- Wenn die Frage auf Deutsch ist, antworte auf Deutsch.
- Wenn die Frage auf Englisch ist, antworte auf Englisch.
- Verwende einfache, verständliche Sprache.
- Vermeide unnötige Fachbegriffe. Wenn Fachbegriffe nötig sind, erkläre sie kurz.

5. Antwortstruktur
- Gib zuerst eine direkte Antwort auf die Frage.
- Danach kannst du wichtige Details als kurze Aufzählung nennen.
- Bei komplexen Fragen darfst du die Antwort strukturieren mit:
  "Kurzantwort", "Details", "Einschränkung" oder "Gefundene Informationen".
- Halte die Antwort informativ, aber nicht unnötig lang.
- Bei einfachen Fragen reichen 2–5 Sätze.
- Bei Analysefragen darf die Antwort ausführlicher sein.

6. Quellenbezug
- Der Kontext enthält Quellenangaben wie Seite, Abschnitt und Inhaltstyp.
- Wenn möglich, erwähne am Ende kurz die relevanten Seiten, zum Beispiel:
  "Die relevanten Informationen stammen aus den Seiten 12 und 15."
- Wenn mehrere Quellen genutzt werden, fasse sie zusammen.
- Nenne keine Quelle, die nicht im Kontext enthalten ist.

7. Unsicherheit
- Wenn der Kontext widersprüchlich ist, sage das.
- Wenn Begriffe mehrdeutig sind, erkläre die wahrscheinlichste Bedeutung anhand des Kontexts.
- Wenn die Frage zu allgemein ist, beantworte sie anhand der gefundenen Chunks und sage, dass nur diese Ausschnitte berücksichtigt wurden.
- Wenn eine Antwort aufgrund fehlender Informationen unsicher ist, formuliere vorsichtig.

==================================================
SPEZIELLE REGELN FÜR NACHHALTIGKEITSBERICHTE
==================================================

Achte besonders auf folgende Themen:

- CO2 / CO₂ / carbon emissions / greenhouse gas emissions / GHG emissions
- Scope 1, Scope 2, Scope 3
- NOX / nitrogen oxides
- Energieverbrauch und erneuerbare Energien
- elektrische Fahrzeuge oder Anzahl von Electric Vehicles
- Impact / Auswirkungen
- Risks / Risiken
- Opportunities / Chancen
- Strategy / Strategie
- Actions / Maßnahmen
- Adopted policies / beschlossene oder eingeführte Richtlinien
- Targets / Ziele
- Reduktionsziele, Klimaneutralität, Net Zero, Zieljahre
- Umweltmanagement, Compliance, Governance
- Vergleich zwischen Jahren

Wenn die Frage eines dieser JSON-Felder betrifft:
CO2, NOX, Number_of_Electric_Vehicles, Impact, Risks, Opportunities, Strategy, Actions, Adopted_policies, Targets

Dann antworte besonders kompakt und extrahiere nur die relevanten Informationen für dieses Feld.

Für JSON-ähnliche Extraktionsfragen gilt:
- Antworte nicht mit langen Erklärungen.
- Gib die wichtigsten gefundenen Informationen in 1–4 kurzen Sätzen.
- Wenn möglich, nenne Zahlen, Einheiten und Jahre.
- Wenn nichts gefunden wurde, schreibe:
  "Keine eindeutige Information im bereitgestellten Kontext gefunden."
- Nutze keine Markdown-Tabelle, außer der Nutzer fragt ausdrücklich danach.

==================================================
BEISPIELE FÜR GUTES VERHALTEN
==================================================

Beispiel 1:
Frage: "Wie hat sich der CO2-Ausstoß entwickelt?"
Gute Antwort:
"Im bereitgestellten Kontext wird ein CO2-Wert für 2023 und 2024 genannt. Der Bericht zeigt, dass sich die Emissionen von X auf Y verändert haben. Damit sind die Emissionen gesunken/gestiegen. Die Information stammt aus Seite Z."

Nur so antworten, wenn X, Y und Z wirklich im Kontext stehen.

Beispiel 2:
Frage: "Welche Risiken nennt der Bericht?"
Gute Antwort:
"Der Bericht nennt im bereitgestellten Kontext folgende Risiken: ... . Weitere Risiken konnte ich im bereitgestellten Kontext nicht eindeutig finden."

Beispiel 3:
Frage: "Wie viele elektrische Fahrzeuge hat das Unternehmen?"
Gute Antwort:
"Die Anzahl elektrischer Fahrzeuge konnte ich im bereitgestellten Kontext nicht eindeutig finden."

Beispiel 4:
Frage: "Fasse die Strategie zusammen."
Gute Antwort:
"Die Strategie konzentriert sich laut Kontext auf ... . Genannt werden außerdem ... . Die Antwort basiert auf den gefundenen Ausschnitten des Berichts."

==================================================
WAS DU NICHT TUN DARFST
==================================================

- Keine Halluzinationen.
- Keine allgemeinen Nachhaltigkeitsdefinitionen, wenn sie nicht gefragt sind.
- Keine erfundenen CO2-Werte.
- Keine erfundenen Zieljahre.
- Keine erfundenen Policies.
- Keine Antwort aus allgemeinem Wissen über das Unternehmen.
- Keine Vermutung als Fakt darstellen.
- Keine Quellenangaben erfinden.
- Keine kompletten langen Kontextpassagen kopieren.
- Keine unnötig langen Antworten bei einfachen Extraktionsfragen.

==================================================
KONTEXT AUS DEM PDF
==================================================

{context}

==================================================
NUTZERFRAGE
==================================================

{question}

==================================================
ANTWORT
==================================================

Bitte beantworte jetzt die oben genannte Nutzerfrage anhand der Regeln oben.

WICHTIG: Antworte vollständig in der Sprache der Nutzerfrage – das gilt für Fließtext,
Struktur-Labels (z.B. "Short Answer" statt "Kurzantwort" bei Englisch) und Quellenangaben.
Diese Regel überschreibt die Sprache dieses Prompts.

"""

    return prompt

def ask(question: str, retriever) -> dict:
    """
    Hauptfunktion: Nimmt eine Frage und einen LangChain-Retriever entgegen,
    sucht relevante Chunks und generiert eine Antwort.

    Args:
        question: Die Frage des Users
        retriever: VectorStoreRetriever aus VectorStore.as_retriever()

    Returns:
        dict mit 'answer' (str) und 'sources' (list von Metadaten)
    """
    # Schritt 1: Relevante Chunks aus ChromaDB holen
    chunks = retriever.invoke(question)

    if not chunks:
        return {
            "answer": "Ich konnte keine relevanten Informationen im Dokument finden.",
            "sources": []
        }

    # Schritt 2: Prompt zusammenbauen
    prompt = build_prompt(question, chunks)

    # Schritt 3: Antwort generieren
    if FAKE_MODE:
        answer = _fake_answer(question, chunks)
    else:
        answer = _call_gwdg_llm(prompt)

    # Schritt 4: Quellen für die UI aufbereiten
    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "section": doc.metadata.get("section", ""),
            "type": doc.metadata.get("type", "text"),
        }
        for doc in chunks
    ]

    return {"answer": answer, "sources": sources}


def _fake_answer(question: str, chunks: list[Document]) -> str:
    """
    Platzhalter-Antwort solange FAKE_MODE = True.
    Zeigt dass das Retrieval funktioniert, ohne echtes LLM.
    """
    preview = chunks[0].page_content[:200] if chunks else ""
    return (
        f"[FAKE MODUS] Das LLM ist noch nicht verbunden.\n\n"
        f"Deine Frage war: '{question}'\n\n"
        f"Das Retrieval hat {len(chunks)} relevante Chunk(s) gefunden. "
        f"Erster Treffer (Vorschau):\n\n\"{preview}...\""
    )


def _call_gwdg_llm(prompt: str) -> str:
    """
    Echter API Call zum GWDG SAIA LLM Server.
    Wird aktiv sobald FAKE_MODE = False.

    Voraussetzung: In ..env muss stehen:
        GWDG_API_KEY=dein_key_hier
        GWDG_BASE_URL=https://saia.gwdg.de/v1  (oder der aktuelle Endpoint)
        GWDG_MODEL=meta-llama-3.1-70b-instruct  (oder gewünschtes Modell)
    """
    try:
        from openai import OpenAI  # GWDG ist OpenAI-kompatibel

        client = OpenAI(
            api_key=os.getenv("GWDG_API_KEY"),
            base_url=os.getenv("GWDG_BASE_URL", "https://saia.gwdg.de/v1"),
        )

        response = client.chat.completions.create(
            model=os.getenv("GWDG_MODEL_NAME", "meta-llama-3.1-70b-instruct"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # niedrig = weniger Halluzinationen
            max_tokens=1024,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Fehler beim LLM-Aufruf: {str(e)}"


# ============================================================
# Zum direkten Testen: python src/rag/rag_chain.py
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Projekt-Root zum Python-Pfad hinzufügen
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.vectorstore.chromadb_storage import VectorStore

    print("=== RAG Chain Test ===")
    print(f"FAKE_MODE: {FAKE_MODE}\n")

    vs = VectorStore()
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    test_frage = "Wie hat sich der CO2-Ausstoß entwickelt?"
    print(f"Testfrage: {test_frage}\n")

    result = ask(test_frage, retriever)

    print(f"Antwort:\n{result['answer']}\n")
    print(f"Quellen: {result['sources']}")
