FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y libopenblas-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
EXPOSE 8501

CMD ["bash", "-c", "uvicorn agent_api:app --host 0.0.0.0 --port 8001 & streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 && wait"]
