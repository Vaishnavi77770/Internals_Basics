FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install pandas

ENTRYPOINT ["python", "src/predict_cli.py"]