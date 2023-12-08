package org.example;

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

public class WineQualityPrediction {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .master("local[*]")
                .getOrCreate();

        LogisticRegressionModel model = LogisticRegressionModel.load("/home/ubuntu/TrainedLogisticRegressionFinal");


        Dataset<Row> newData = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .csv("ValidationDataset.csv");

        String[] originalCols = newData.columns();
        for (String col : originalCols) {
            String newCol = col.replaceAll("\"", "");
            newData = newData.withColumnRenamed(col, newCol);
        }

        String[] inputCols = {
                "fixed acidity", "volatile acidity", "citric acid",
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH",
                "sulphates", "alcohol"
        };

        for (String col : inputCols) {
            newData = newData.withColumn(col, newData.col(col).cast(DataTypes.DoubleType));
        }
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> transformedNewData = assembler.transform(newData);

        Dataset<Row> predictions = model.transform(transformedNewData);
        predictions.show();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("mse");

        double mse = evaluator.evaluate(predictions);
        System.out.println("Mean Squared Error (MSE) on Validation data = " + mse);

        evaluator.setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on Validation data = " + rmse);

        MulticlassClassificationEvaluator evaluator1 = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator1.evaluate(predictions);
        System.out.println("F1 Score on Validation data = " + f1);

        spark.stop();
    }
}
