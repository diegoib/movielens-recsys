FROM python:3.11-slim-bookworm

# Java 17 JRE — PyFlink uses a JVM bridge; no full JDK needed
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless curl && \
    rm -rf /var/lib/apt/lists/*

# PyFlink 1.18.1 is the latest stable release with full PyPI support.
# Flink 2.x connectors are not yet available on Maven Central.
RUN pip install --no-cache-dir apache-flink==1.18.1 redis==5.0.8

# Download the Kafka connector JAR into PyFlink's own lib directory.
# The version suffix (-1.18) must match the installed apache-flink version.
RUN PYFLINK_LIB=$(python3 -c "import pyflink, os; print(os.path.join(os.path.dirname(pyflink.__file__), 'lib'))") && \
    curl -fsSL \
    "https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.1.0-1.18/flink-sql-connector-kafka-3.1.0-1.18.jar" \
    -o "${PYFLINK_LIB}/flink-sql-connector-kafka-3.1.0-1.18.jar"

WORKDIR /app
COPY src/ ./src/
ENV PYTHONPATH=/app

CMD ["python3", "src/features/processor.py"]
