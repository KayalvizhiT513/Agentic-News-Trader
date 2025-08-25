FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir requests
COPY fetch.py .
CMD ["python","fetch.py"]
