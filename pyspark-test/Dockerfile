FROM bitnami/spark:latest

WORKDIR /app

COPY test_pyspark.py .

CMD ["spark-submit", "test_pyspark.py"]
