# API image (CPU, surrogate or small jobs). For GPU search use Dockerfile.worker.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV RLNAS_EVALUATOR=surrogate RLNAS_MAX_BUDGET=100
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
