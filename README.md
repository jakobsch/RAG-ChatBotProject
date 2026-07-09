# 🌱 Sustainability Report RAG ChatBot

A web-based Retrieval Augmented Generation (RAG) application that enables users to upload Environmental Sustainability Reports (PDF) from large companies and interactively ask questions about their content.

---

### 📦 Benutzung

---

### 🛠️ Schritte zum Funktionieren (lokal, ohne Docker)

1. **Repository klonen**
   Klone unser Git-Repository in einen gewünschten Ordner und navigiere anschließend in diesen Ordner:

   ```bash
   cd "folder_name"
   ```

2. **Virtual Environment erstellen und aktivieren**
   Führe im Terminal (im Projektordner) folgende Befehle aus.

   **Windows (PowerShell):**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Abhängigkeiten installieren**

   ```bash
   pip install -r requirements.txt
   ```

4. **.env-Datei erstellen**
   Im Repository liegt eine Vorlage `.env.example`. Kopiere sie zu `.env` und trage deinen eigenen GWDG-API-Key ein. Die Datei sieht so aus:

   ```env
   GWDG_API_KEY=your_api_key_here
   GWDG_BASE_URL=https://chat-ai.academiccloud.de/v1
   GWDG_MODEL_NAME=meta-llama-3.1-70b-instruct
   
   HF_EMBEDDING_MODEL=all-MiniLM-L6-v2
   HF_TOKEN=your_huggingface_api_token
   EMBEDDING_PROVIDER= "huggingface" oder "gwdg"
   GWDG_EMBEDDING_MODEL=e5-mistral-7b-instruct
   
   
   ```

   Speichere anschließend die Datei.

   > **Hinweis:** Die `.env` mit dem echten Key ist bewusst nicht im Repository (sie steht in der `.gitignore`). Jede:r muss also die eigene `.env` mit einem gültigen Key anlegen.

---

### 🚀 Schritte zum Ausführen (aktuelle Nutzung)

1. Im Terminal in den Projektordner navigieren.
2. Virtual Environment aktivieren (falls noch nicht aktiv) und die App starten:

   **Windows (PowerShell):**

   ```bash
   .venv\Scripts\activate
   streamlit run app.py
   ```

   **macOS / Linux:**

   ```bash
   source .venv/bin/activate
   streamlit run app.py
   ```

Die App ist danach im Browser unter `http://localhost:8501` erreichbar.

---

### 🔁 Wiederholte Benutzung

Wenn du das Projekt erneut öffnest, musst du nur das Virtual Environment neu aktivieren und die App starten:

**Windows (PowerShell):**

```bash
.venv\Scripts\activate
streamlit run app.py
```

**macOS / Linux:**

```bash
source .venv/bin/activate
streamlit run app.py
```

---

### 🐳 Nutzung mit Docker

Alternativ kann das Projekt containerisiert ausgeführt werden. Es wird nur Docker benötigt (kein docker-compose).

#### Voraussetzungen

* Docker (unter Windows: Docker Desktop mit aktiviertem WSL2)

#### Schritte

1. **.env-Datei anlegen**
   Wie oben beschrieben: `.env.example` zu `.env` kopieren und den eigenen GWDG-API-Key eintragen. Der Key wird beim Start in den Container gereicht und ist **nicht** im Image enthalten.

2. **Docker Image bauen**
   Im Projektordner ausführen:

   ```bash
   docker build -t rag-chatbot .
   ```

3. **Container starten**

   **Windows (PowerShell):**

   ```bash
   docker run --env-file .env -p 8501:8501 -v "${PWD}/data:/app/data" rag-chatbot
   ```

   **macOS / Linux:**

   ```bash
   docker run --env-file .env -p 8501:8501 -v "$(pwd)/data:/app/data" rag-chatbot
   ```

   Dabei wird:

   * der GWDG-Key über `--env-file .env` sicher in den Container gereicht,
   * Port `8501` nach außen geöffnet,
   * die ChromaDB über ein Volume (`data`-Ordner) persistent gehalten, sodass verarbeitete PDFs einen Neustart des Containers überstehen.

4. **Streamlit App aufrufen**
   Im Browser öffnen:

   ```
   http://localhost:8501
   ```

5. **Container beenden**
   Zum Beenden im Terminal `Ctrl + C` drücken.

---

### 🧩 Tech-Stack

* **Frontend:** Streamlit (inkl. eingebautem PDF-Viewer mit Sprung zur Quellseite)
* **PDF-Parsing:** pypdf
* **Chunking:** LangChain `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200)
* **Embeddings:** `all-MiniLM-L6-v2` (384 Dimensionen), lokal
* **Vektor-Datenbank:** ChromaDB (persistent, on disk)
* **Retrieval:** Hybrid Search (BM25 + Vektorsuche, per Ensemble fusioniert) mit anschließendem Cross-Encoder-Reranking (`ms-marco-MiniLM-L-6-v2`)
* **LLM:** Llama 3.1 70B über die GWDG API (OpenAI-kompatibel)

---

### 🗂️ Projektstruktur

* `app.py` – Streamlit-Oberfläche und Steuerung (Upload, Chat, PDF-Viewer)
* `src/pdf_processing/ingestion.py` – PDF einlesen und in Chunks aufteilen
* `src/vectorstore/chromadb_storage.py` – Chunks speichern und die relevanten Stellen suchen (Hybrid Search + Reranking)
* `src/rag/rag_chain.py` – Prompt bauen und die Anfrage an das LLM stellen

---

### 🗂️ Trello

Link to the Trello Board: https://trello.com/b/uKnRBWEo/rag-chatbot-application

