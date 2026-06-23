# step 13 — fastapi service image
# note: base bumped to 3.11-slim (not 3.10) because scikit-learn==1.9.0
# has no python 3.10 wheel — needed to match the version the model
# was actually trained and pickled with
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]