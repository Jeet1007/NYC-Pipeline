import sys
import logging
import boto3
import json
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window

#=================Logging================

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#=================Job Parameters================

args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "bucket",
    "silver_path",
    "gold_path"
])

bucket = args["bucket"]
silver_path = args["silver_path"]
gold_path = args["gold_path"]

logger.info(f"Starting Gold Transformation!")

#================Boto3 Client====================

s3_client = boto3.client("s3")

#===============Spark/Glue Context================

sc = SparkContext()
gc = GlueContext(sc)
spark = gc.spark_session
job = Job(gc)
job.init(args["JOB_NAME"], args)

#================Check if File exists===============

def file_exists(bucket, key):
    response = s3_client.list_objects_v2(
        Bucket = bucket,
        Prefix = key,
        MaxKeys = 1
    )
    if "Contents" not in response:
        logger.warning(f"File does not exist in s3://{bucket}/{key}. ")
        return False
    return True

#==================Read silver=================

def read_silver_pq(bucket, key):
    prefix = f"{silver_path}/{key}"
    path = f"s3://{bucket}/{prefix}"
    if not file_exists(bucket, prefix):
        raise FileNotFoundError(
            f"File does not exist at {path}!"
        )
    return spark.read.parquet(path)

def read_silver_csv(bucket, key):
    prefix = f"{silver_path}/{key}"
    path = f"s3://{bucket}/{prefix}"
    if not file_exists(bucket, prefix):
        raise FileNotFoundError(
            f"File does not exist at {path}!"
        )
    return spark.read.option("inferSchema", "true").option("header", "true").csv(path)

#===============Write parquet================

def write_pq(df, bucket, key):
    path = f"s3://{bucket}/{gold_path}/{key}"
    df.coalesce(1).write.mode("overwrite").parquet(path)

#================Creating joined facts and dimensions==================
"""
Here we join the Silver Facts with the lookup(static for now) to get a holistic, single source of truth for our data.
"""
df_silver = read_silver_pq(bucket, "facts_silver")
df_lookup = read_silver_csv(bucket, "zone_lookup")

df_gold = df_silver.join(df_lookup, df_silver.PULocationID == df_lookup.LocationID, how="left")
df_gold = df_gold.withColumnRenamed("Borough", "PUBorough")\
                    .withColumnRenamed("Zone", "PUZone")\
                    .withColumnRenamed("service_zone", "PUservice_zone")
df_gold = df_gold.drop("LocationID")

df_gold = df_gold.join(df_lookup, df_gold.DOLocationID == df_lookup.LocationID, how="left")
df_gold = df_gold.withColumnRenamed("Borough", "DOBorough")\
                    .withColumnRenamed("Zone", "DOZone")\
                    .withColumnRenamed("service_zone", "DOservice_zone")
df_gold = df_gold.drop("LocationID")
write_pq(df_gold, bucket, "full")

#===============Creating aggregated tables for business questions===================

# 1.Avg Fare/Distance per PU borough
df_gold1 = df_gold.withColumn("fare_per_unit_distance", F.col("fare_amount")/F.col("trip_distance"))
df_gold1 = df_gold1.groupBy("PUBorough").agg(F.avg("fare_per_unit_distance").alias("avg_fare_per_mile"))
write_pq(df_gold1, bucket, "avg_fare_per_mile")

# Demand and Volume:
## 1. Total trip count by zone
df_gold2 = df_gold.groupBy("PUZone").agg(F.count("*").alias("total_trip_count_per_zone"))
write_pq(df_gold2, bucket, "total_trips_count_per_zone")

## 2. Number of trips from a specific PUZone at a specific PU hour, i.e, number of trips per PUZone per hour
df_gold3 = df_gold.withColumn("pickup_hour", F.hour(F.col("tpep_pickup_datetime")))
df_gold3 = df_gold3.groupBy("PUZone", "pickup_hour").agg(
    F.count("*").alias("trip_per_zone_per_hour")
)
write_pq(df_gold3, bucket, "trips_per_zone_per_hour")

## 3. Month by Month trip value trends
df_gold4 = df_gold.withColumn("pickup_month", F.month(F.col("tpep_pickup_datetime")))
df_gold4 = df_gold4.groupBy("pickup_month").agg(F.count("*").alias("trips_per_month"))
write_pq(df_gold4, bucket, "trips_per_month")

## 4. Zone-wise Peak hours. Taking top three hours with most trips for each zone 
windowSpec1 = Window.partitionBy(F.col("PUZone")).orderBy(F.col("trip_per_zone_per_hour").desc())
df_gold5 = df_gold3.withColumn("row_num", F.row_number().over(windowSpec1))
df_gold5 = df_gold5.filter(F.col("row_num") < 4)
write_pq(df_gold5, bucket, "zone_wise_peak_hours")

#==================Job Commit==================
job.commit()
logger.info("Job 3 is completed!")