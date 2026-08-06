import sys
import json
import logging
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F


#===============Logging================

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#===============Job Parameters==============

args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "year",
    "month",
    "bucket",
    "raw_path",
    "silver_path"
])

year = args["year"]
month = args["month"]
bucket = args["bucket"]
raw_path = args["raw_path"]
silver_path = args["silver_path"]

logger.info(f"Starting silver transformation for year={year}, month={month}")

#===================Boto3 Client=================

s3_client = boto3.client("s3")

#===================Spark/Glue Context=================

sc = SparkContext()
gc = GlueContext(sc)
spark = gc.spark_session
job = Job(gc)
job.init(args["JOB_NAME"], args)

#==================Check if file exists in specified S3 location===================

def file_exists(bucket, key):
    response = s3_client.list_objects_v2(
        Bucket = bucket,
        Prefix = key,
        MaxKeys = 1
    )
    if "Contents" not in response:
        logger.warning(f"File does not exist in s3://{bucket}/{key}")
        return False
    return True

#=================Read Partitioned Data from Raw======================

def read_partitioned(bucket, year, month):
    key = f"{raw_path}/facts/year={year}/month={month}"
    path = f"s3://{bucket}/{key}"
    if not file_exists(bucket, key):
        raise FileNotFoundError(
            f"File not found in {path}"
        )
    return spark.read.parquet(path)

#==================Schema Validation==========================
"""
Check the schema for addition, renaming, removal of columns. 
If there is an addition schema evolution should handle it on write.
If there is a renaming/removal of a column, Fail loud and alert the Developer.
"""

df_silver = read_partitioned(bucket, year, month)

req_cols = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID",
            "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
            "improvement_surcharge", "total_amount", "congestion_surcharge", "Airport_fee", "cbd_congestion_fee", "ingestion_timestamp", "source_file_name"]

curr_cols = df_silver.columns
missing_cols = set(req_cols) - set(curr_cols)

if missing_cols:
    raise ValueError(f"Missing Required Columns: {missing_cols}")

added_cols = set(curr_cols) - set(req_cols)
logger.info(f"Added columns are: {added_cols}")

#==================Data Checks=====================

write_path = f"s3://{bucket}/{silver_path}"
rejected_write_path = f"s3://{bucket}/Rejected/silver_dropped"

zone_lookup_path = f"s3://{bucket}/{silver_path}/zone_lookup"
df_zone_lookup = spark.read.option("header", "true").option("inferSchema", "true").csv(zone_lookup_path)

# 1. Check if tpep_dropoff_datetime - tpep_pickup_datetime > 0
df_silver = df_silver.withColumn(
    "timediff", F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")
)
df_silver_faulty_datetimes = df_silver.filter(F.col("timediff") < 0)
df_silver = df_silver.subtract(df_silver_faulty_datetimes)
df_silver = df_silver.drop("timediff")
df_silver_faulty_datetime = df_silver_faulty_datetimes.drop("timediff")
df_silver_faulty_datetime.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/faulty_timestamps/year={year}/month={month}")

# 2. Check if passenger_count = 0
df_silver_faulty_passenger_count = df_silver.filter(F.col("passenger_count") == 0)
df_silver = df_silver.subtract(df_silver_faulty_passenger_count)
df_silver_faulty_passenger_count.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/faulty_passenger_count/year={year}/month={month}")

# 3. Check if trip_distance = 0
df_silver_faulty_trip_distance = df_silver.filter(F.col("trip_distance") == 0)
df_silver = df_silver.subtract(df_silver_faulty_trip_distance)
df_silver_faulty_trip_distance.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/faulty_trip_distance/year={year}/month={month}")

# 4. PULocationID and DULocatioID have some corresponding values in lookup tables
"""
For Intial Prototype we are having a Static Lookup table. 
Here we will left join(broadcast) df_silver on df_lookup to get the valid data twice, once for PULocationId, once for DOLocationID 
Once SCD2 is maintained we will add one more condititon that the status of the record in df_lookup should be Active
"""
df_joined_pickup_locations = df_silver.join(df_zone_lookup, df_silver.PULocationID == df_zone_lookup.LocationID, how="left")
df_joined_dropoff_locations = df_silver.join(df_zone_lookup, df_silver.DOLocationID == df_zone_lookup.LocationID, how="left")
 
df_faulty_pickup_locations = df_joined_pickup_locations.filter(F.col("Zone").isNull())
df_faulty_dropoff_locations = df_joined_dropoff_locations.filter(F.col("Zone").isNull())

df_faulty_pickup_locations = df_faulty_pickup_locations.select(df_silver.columns)
df_faulty_dropoff_locations = df_faulty_dropoff_locations.select(df_silver.columns)

df_silver = df_silver.subtract(df_faulty_pickup_locations)
df_silver = df_silver.subtract(df_faulty_dropoff_locations)

df_faulty_pickup_locations.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/faulty_pickup_locations/year={year}/month={month}")
df_faulty_dropoff_locations.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/faulty_dropoff_locations/year={year}/month={month}")

#=======================Writing Silver DataFrame in Parquet format in append mode======================

df_silver.coalesce(1).write.mode("append").parquet(f"{write_path}/facts")
# df_silver.show(20)
# print(df_silver.count())

# month=01 + month=02
# df_silver1 = spark.read.parquet(f"s3://{bucket}/silver/facts")
# print(df_silver1.count())

#=================Job Commit===================

job.commit()
logger.info("Job 2 completed")