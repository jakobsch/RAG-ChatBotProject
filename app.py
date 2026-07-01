"""
app.py  –  Streamlit Chat UI für den RAG-Chatbot
Starten mit: streamlit run app.py
"""

import sys
import json
import base64
import tempfile
from pathlib import Path

import streamlit as st

# Projekt-Root zum Python-Pfad hinzufügen damit src-Importe funktionieren
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pdf_processing.ingestion import load_and_chunk
from src.vectorstore.chromadb_storage import VectorStore
from src.rag.rag_chain import ask
from streamlit_pdf_viewer import pdf_viewer

# ── Seitenkonfiguration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sustainability Report Chatbot",
    page_icon="🌱",
    layout="wide",
)

# ── Session State initialisieren ───────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None          # NEU: PDF für Viewer speichern
if "processing" not in st.session_state:
    st.session_state.processing = False
if "viewer_page" not in st.session_state:
    st.session_state.viewer_page = None        # NEU: angeklickte Quell-Seite

# Farbpalette
BG         = "#f4f7f4"
BOT_BG     = "#ffffff"
BOT_TEXT   = "#1a2e1a"
BOT_BORDER = "#d4e6d4"
BADGE_BG   = "#e8f4e8"
BADGE_TEXT = "#2d5a2d"
BADGE_BRD  = "#b0d4b0"
STATUS_BG  = "#e8f4e8"
WARN_BG    = "#fff8e1"
WARN_TEXT  = "#5d4037"
H_COLOR    = "#1a2e1a"
ACCENT     = "#2d5a2d"
ACCENT_H   = "#4a8c4a"
# css
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
  }}
  h1, h2, h3 {{
    font-family: 'DM Serif Display', serif;
    color: {H_COLOR} !important;
  }}

  .stApp {{ background-color: {BG}; }}

   [data-testid="stSidebar"] {{
    background-color: #1a2e1a;
    color: #e8f0e8;
  }}
  [data-testid="stSidebar"] * {{ color: #e8f0e8 !important; }}

  
 /* Uploader-Inhalt: mittleres Grau, lesbar auf weiss UND schwarz */
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
  [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] * {{
    color: #888888 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg,
  [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg {{
    fill: #888888 !important;
  }}
  [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {{
    stroke: #888888 !important;
    fill: none !important;
    color: #888888 !important;
  }}
  [data-testid="stSidebar"] .stButton > button {{
    background-color: #2d5a2d;
    color: #e8f0e8;
    border: 1px solid #4a8c4a;
    border-radius: 8px;
    width: 100%;
  }}
  [data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #4a8c4a;
  }}

  .chat-user {{
    background: {ACCENT};
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 15%;
    font-size: 0.95rem;
    line-height: 1.5;
  }}
  .chat-bot {{
    background: {BOT_BG};
    color: {BOT_TEXT};
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 15% 8px 0;
    font-size: 0.95rem;
    line-height: 1.5;
    border: 1px solid {BOT_BORDER};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .source-badge {{
    display: inline-block;
    background: {BADGE_BG};
    color: {BADGE_TEXT};
    border: 1px solid {BADGE_BRD};
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 4px 4px 0 0;
    cursor: pointer;
  }}
  .status-box {{
    background: {STATUS_BG};
    border-left: 4px solid {ACCENT};
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: {BOT_TEXT};
    margin-bottom: 12px;
  }}
  .fake-warning {{
    background: {WARN_BG};
    border-left: 4px solid #f9a825;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: {WARN_TEXT};
    margin-bottom: 12px;
  }}
  /* NEU: Quellen-Buttons als Badges stylen */
  div[data-testid="stHorizontalBlock"] .stButton > button {{
    background-color: {BADGE_BG} !important;
    color: {BADGE_TEXT} !important;
    border: 1px solid {BADGE_BRD} !important;
    border-radius: 12px !important;
    font-size: 0.75rem !important;
    padding: 2px 10px !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: PDF Upload ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 Sustainability\nReport Chatbot")

    st.markdown("---")
    st.markdown("### PDF hochladen")

    uploaded_file = st.file_uploader(
        "Nachhaltigkeitsbericht auswählen",
        type=["pdf"],
        help="Nur PDF-Dateien werden unterstützt.",
    )

    if uploaded_file is not None:
        if st.button("📄 PDF verarbeiten"):
            with st.spinner("PDF wird verarbeitet..."):
                try:
                    vs = VectorStore()
                    already_stored = vs.is_document_stored(uploaded_file.name)

                    pdf_bytes = uploaded_file.read()

                    if already_stored:
                        st.info(f"ℹ️ '{uploaded_file.name}' ist bereits in der DB – Chunking wird übersprungen.")
                        vs.load_documents_for_bm25(uploaded_file.name)
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_bytes)
                            tmp_path = tmp.name

                        chunks = load_and_chunk(tmp_path)

                        # Echten Dateinamen statt Temp-Pfad in die Metadaten schreiben
                        for chunk in chunks:
                            chunk.metadata["filename"] = uploaded_file.name

                        vs.save_documents_to_db(chunks)

                        st.success(f"✅ {len(chunks)} Chunks gespeichert!")

                    st.session_state.retriever = vs.as_hybrid_reranking_retriever(
                        filename=uploaded_file.name, k=4
                    )

                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_bytes = pdf_bytes
                    st.session_state.chat_history = []
                    st.session_state.viewer_page = None

                except Exception as e:
                    st.error(f"Fehler beim Verarbeiten: {e}")

    if st.session_state.pdf_name:
        st.markdown(
            f'<div class="status-box">📑 <b>{st.session_state.pdf_name}</b><br>'
            f"ist geladen und bereit.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button("🗑️ Chat leeren"):
        st.session_state.chat_history = []
        st.session_state.viewer_page = None
        st.rerun()

    st.markdown("### JSON Export")
    st.caption("Nach dem Laden eines PDFs können Key-Daten exportiert werden.")
    if st.button("📥 JSON generieren & herunterladen"):
        if st.session_state.retriever is None:
            st.warning("Bitte zuerst ein PDF hochladen.")
        else:
            with st.spinner("Extrahiere Key-Daten..."):
                felder = [
                    "CO2", "NOX", "Number_of_Electric_Vehicles",
                    "Impact", "Risks", "Opportunities",
                    "Strategy", "Actions", "Adopted_policies", "Targets"
                ]
                json_data = {"name": st.session_state.pdf_name}
                for feld in felder:
                    result = ask(
                        f"Was steht im Bericht über {feld}? Kurze Zusammenfassung.",
                        st.session_state.retriever,
                    )
                    json_data[feld] = result["answer"]

                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ JSON herunterladen",
                    data=json_str,
                    file_name=f"{st.session_state.pdf_name}_extract.json",
                    mime="application/json",
                )

# ── Hauptbereich: Chat ─────────────────────────────────────────────────────────
st.markdown("# Frag deinen Nachhaltigkeitsbericht")

from src.rag.rag_chain import FAKE_MODE
if FAKE_MODE:
    st.markdown(
        '<div class="fake-warning">⚠️ <b>Fake-Modus aktiv</b> – '
        "Das LLM ist noch nicht verbunden. Antworten sind Platzhalter. "
        "Sobald der GWDG API Key eingetragen ist, einfach <code>FAKE_MODE = False</code> "
        "in <code>src/rag/rag_chain.py</code> setzen.</div>",
        unsafe_allow_html=True,
    )

if st.session_state.retriever is None:
    st.info("👈 Lade zuerst einen Nachhaltigkeitsbericht in der Sidebar hoch.")
else:
    # NEU: zwei Spalten wenn PDF-Viewer aktiv
    show_viewer = bool(st.session_state.viewer_page and st.session_state.pdf_bytes)
    if show_viewer:
        chat_col, viewer_col = st.columns([3, 2])
    else:
        chat_col = st.container()
        viewer_col = None

    with chat_col:
        # Chat-Verlauf anzeigen
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )
                # NEU: Quellen als klickbare Buttons statt statische Badges
                if msg.get("sources"):
                    # Doppelte Seiten entfernen, Reihenfolge behalten
                    seen = set()
                    unique_sources = []
                    for s in msg["sources"]:
                        if s["page"] not in seen:
                            seen.add(s["page"])
                            unique_sources.append(s)

                    cols = st.columns(len(unique_sources))
                    for j, (col, s) in enumerate(zip(cols, unique_sources)):
                        with col:
                            label = f'Seite {s["page"]} · {s["section"][:20]}'
                            if st.button(label, key=f"src_{i}_{j}"):
                                st.session_state.viewer_page = s["page"]
                                st.rerun()

        # NEU: Eingabefeld in st.form → Enter zum Senden + Feld leert sich
        st.markdown("---")
        with st.form(key="question_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                user_input = st.text_input(
                    "Deine Frage",
                    placeholder="z.B. Wie hat sich der CO2-Ausstoß zwischen 2023 und 2024 verändert?",
                    label_visibility="collapsed",
                )
            with col2:
                send = st.form_submit_button("Senden", use_container_width=True)

        if send and user_input.strip():
            st.session_state.chat_history.append(
                {"role": "user", "text": user_input}
            )
            with st.spinner("Suche relevante Stellen..."):
                result = ask(user_input, st.session_state.retriever)
            st.session_state.chat_history.append(
                {
                    "role": "bot",
                    "text": result["answer"],
                    "sources": result["sources"],
                }
            )
            st.rerun()

    # NEU: PDF-Viewer rechts
    if show_viewer:
        with viewer_col:
            close1, close2 = st.columns([5, 1])
            with close1:
                st.markdown(f"#### 📄 Seite {st.session_state.viewer_page}")
            with close2:
                if st.button("✕", key="close_viewer"):
                    st.session_state.viewer_page = None
                    st.rerun()

            page = st.session_state.viewer_page
            pdf_viewer(
                st.session_state.pdf_bytes,
                width=700,
                height=650,
                scroll_to_page=page,
                key=f"pdfview_{page}",
            )