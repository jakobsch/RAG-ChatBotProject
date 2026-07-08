# 1. Basis-Image: ein schlankes Linux mit Python 3.11 vorinstalliert
FROM python:3.11-slim

# 2. Arbeitsverzeichnis im Container festlegen
WORKDIR /app

# 3. Zuerst NUR die requirements kopieren (nicht den ganzen Code)
COPY requirements.txt .

# 4. Die Python-Pakete installieren
RUN pip install --no-cache-dir -r requirements.txt

# 5. Jetzt den restlichen Code reinkopieren
COPY . .

# 6. Den Streamlit-Port nach außen ankündigen
EXPOSE 8501

# 7. Der Startbefehl, wenn der Container läuft
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]