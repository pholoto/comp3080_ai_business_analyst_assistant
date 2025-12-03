FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY back_end/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY AI ./AI
COPY common ./common

EXPOSE 8000

CMD ["uvicorn", "AI.app:app", "--host", "0.0.0.0", "--port", "8000"]