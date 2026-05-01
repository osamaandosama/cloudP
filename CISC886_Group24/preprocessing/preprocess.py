"""
CISC 886 — Cloud Computing — Group 24
Smart Banking AI — Preprocessing pipeline (Apache Spark on Amazon EMR)

This script reads the raw Bitext retail-banking CSV from S3,
drops null rows (sanity check), splits the cleaned dataframe
80/10/10 (train/val/test) with a fixed seed, and writes three
Parquet files back to S3 under cleaned-data/.

Submitted as an EMR Step:

    command-runner.jar spark-submit --deploy-mode cluster \
        s3://25fwmh-bank-project/scripts/preprocess.py
"""

from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("G24-DataPreprocessing")
        .getOrCreate()
    )

    bucket = "s3://25fwmh-bank-project"
    input_path = f"{bucket}/input/bitext-retail-banking-llm-chatbot-training-dataset.csv"
    output_path = f"{bucket}/cleaned-data"

    # 1) Load the raw CSV.  multiLine + escape='"' are needed because
    #    the response field contains embedded commas and quotes.
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(input_path)
    )

    # 2) Sanity-clean: drop any row with a null cell.
    cleaned_df = df.dropna()

    # 3) Reproducible 80/10/10 split.  Same seed is used in the
    #    Colab notebook (Cell 5), so train/val/test rows are
    #    identical between preprocessing and fine-tuning -- no
    #    leakage is possible across the boundary.
    train, val_test = cleaned_df.randomSplit([0.8, 0.2], seed=42)
    val, test = val_test.randomSplit([0.5, 0.5], seed=42)

    # 4) Write three single-file Parquet outputs.  coalesce(1)
    #    keeps the EMR-side artefact small and easy to download.
    train.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/train.parquet")
    val.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/val.parquet")
    test.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/test.parquet")

    spark.stop()


if __name__ == "__main__":
    main()
