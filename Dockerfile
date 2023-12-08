# Use a base image with Java 11
FROM openjdk:11-jdk

# Install Spark (adjust the Spark version and URL as needed)
RUN apt-get update && apt-get install -y wget
RUN wget https://dlcdn.apache.org/spark/spark-3.3.3/spark-3.3.3-bin-hadoop3.tgz
RUN tar -xzf spark-3.3.3-bin-hadoop3.tgz && \
    mv spark-3.3.3-bin-hadoop3 /usr/local/spark
ENV SPARK_HOME /usr/local/spark

# Copy your application JAR
COPY ./target/prediction-cloud-computing-1.0-SNAPSHOT.jar /app/

COPY ./src/main/resources/TrainedLogisticRegressionFinal /app/src/main/resources/TrainedLogisticRegressionFinal
COPY ./src/main/resources/ValidationDataset.csv /app/src/main/resources/ValidationDataset.csv


# Set the working directory and entry point
WORKDIR /app
ENTRYPOINT ["/usr/local/spark/bin/spark-submit", "--class", "org.example.WineQualityPrediction", "--master", "local[*]", "/app/prediction-cloud-computing-1.0-SNAPSHOT.jar"]
