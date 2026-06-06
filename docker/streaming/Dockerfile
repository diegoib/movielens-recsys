FROM flink:2.2.0-scala_2.12-java17

# Install Python 3 + redis client (PyFlink ships via the JVM, redis is the only Python dep)
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir apache-flink==1.18.1 redis==5.0.8

# Kafka connector JAR — needed for KafkaSource API (RedPanda is Kafka-compatible)
ADD https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.1.0-1.18/flink-sql-connector-kafka-3.1.0-1.18.jar \
    /opt/flink/lib/

WORKDIR /app
COPY src/ ./src/
ENV PYTHONPATH=/app

CMD ["python3", "src/features/processor.py"]
