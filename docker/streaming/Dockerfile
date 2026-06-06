FROM flink:2.2.0-scala_2.12-java17

# Flink 2.2.0 base image is Debian bookworm.
# - python3-venv is NOT in the default bookworm repos → removed.
# - pip requires --break-system-packages on Debian 12+ (externally-managed-environment).
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# apache-flink pip version must match the base image Flink version.
# Flink 2.x PyPI package: apache-flink>=2.0.0 (1.x packages are incompatible with Flink 2.x JVM).
RUN pip3 install --no-cache-dir --break-system-packages apache-flink>=2.0.0 redis==5.0.8

# Kafka connector JAR for Flink 2.x.
# Note: the connector artifact versioning for Flink 2.x uses the pattern flink-sql-connector-kafka-<connector-version>.jar
ADD https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.3.0-2.0/flink-sql-connector-kafka-3.3.0-2.0.jar \
    /opt/flink/lib/

WORKDIR /app
COPY src/ ./src/
ENV PYTHONPATH=/app

CMD ["python3", "src/features/processor.py"]
