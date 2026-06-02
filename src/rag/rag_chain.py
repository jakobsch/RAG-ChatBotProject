"""
src/rag/rag_chain.py

Baut die RAG-Chain: Frage + relevante Chunks -> Antwort vom LLM.

Aktuell: FAKE_MODE = True  -> kein echter API-Call, Platzhalter-Antwort
Sobald API Key vorhanden: FAKE_MODE = False setzen und GWDG-Zugangsdaten in .env eintragen
"""

import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional
from langchain_core.documents import Document

# ============================================================
# FAKE MODE SCHALTER
# True  = kein LLM nötig, gibt Platzhalter-Antwort zurück
# False = echter GWDG API Call (braucht API Key in .env)
# ============================================================
FAKE_MODE = False


def build_prompt(question: str, chunks: list[Document]) -> str:
    """
    Baut den Prompt aus Frage + Chunks zusammen.
    Der LLM bekommt nur die relevanten Textstellen,
    damit er keine halluzinierten Antworten produziert.
    """
    context = "\n\n---\n\n".join([doc.page_content for doc in chunks])

    prompt = f"""Du bist ein hilfreicher Assistent der Fragen zu Nachhaltigkeitsberichten beantwortet.
Antworte ausschließlich basierend auf dem folgenden Kontext. 
Wenn die Antwort nicht im Kontext zu finden ist, sage das ehrlich.

Kontext:
{context}

Frage: {question}

Antwort:"""
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

    Voraussetzung: In .env muss stehen:
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
