FROM python:3.11-slim

WORKDIR /app

# Installer les dependances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code et les artefacts necessaires
COPY app/ ./app/
COPY model/ ./model/
COPY data/sample_clients.csv ./data/sample_clients.csv
COPY data/reference_data.csv ./data/reference_data.csv

# Creer le repertoire de logs
RUN mkdir -p logs

EXPOSE 7860

CMD ["python", "-m", "app.app"]
