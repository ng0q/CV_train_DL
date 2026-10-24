FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU wheels: the API loads the checkpoint with map_location="cpu".
RUN grep -Ev '^(torch|torchvision)==' requirements.txt > requirements.runtime.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.1 \
    && pip install --no-cache-dir -r requirements.runtime.txt \
    && rm requirements.runtime.txt

COPY app/ ./app/
COPY model/model.pth ./model/model.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
