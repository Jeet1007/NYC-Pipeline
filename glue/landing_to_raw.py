import sys
import json
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

#=================Logging=====================

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#=================Job Parameters=================
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "year",
    "month",
    "bucket",
    "landing_path",
    "raw_path"
])

year = args["year"]
month = args["month"]
bucket = args["bucket"]
landing_path = args["landing_path"]
raw_path = args["raw_path"]

logger.info(f"Starting Job1 for year={year}, month={month}")

#================Boto3 Clients==================

s3_client = boto3.client("s3")

#================Spark/Glue context============

sc = SparkContext()
gc = GlueContext(sc)
spark = gc.spark_session
job = Job(gc)
job.init(args["JOB_NAME"], args)

#=================Check if file exists in the specified S3 path=====================

def file_exists(bucket, key):
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=key,
        MaxKeys=1
    )
    if "Contents" not in response:
        logger.warning(f"File does not exit: s3://{bucket}/{key}")
        return False
    return True
    

#=================Read Partitioned Data from Landing Zone=====================

def read_partitioned(bucket, year, month):
    key = f"{landing_path}/yellow_tripdata_{year}-{month}.parquet"
    path = f"s3://{bucket}/{key}"
    check = file_exists(bucket, key)
    if check == False:
        raise FileNotFoundError(
            f"File not found in s3://{bucket}/{key}"
        )
        
    return spark.read.parquet(path)

#==================Cleaning the landed file======================

col_timestamps = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
col_locations = ["PULocationID", "DOLocationID"]

df_lz = read_partitioned(bucket, year, month)

req_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
cols = df_lz.columns

missing_cols = set(req_cols) - set(cols)
if missing_cols:
    raise ValueError(f"Missing Required Columns: {missing_cols}")

df_lz_count = df_lz.count()
rejected_write_path =  f"s3://{bucket}/Rejected"
write_path = f"s3://{bucket}/{raw_path}"

# 1. Deduping
df_lz_deduped = df_lz.dropDuplicates()
df_lz_deduped_count = df_lz_deduped.count()
df_rejected_deduped = df_lz.exceptAll(df_lz_deduped)
df_rejected_deduped.coalesce(1).write.mode('overwrite').parquet(f"{rejected_write_path}/raw_dropped/dropped_duplicates/year={year}/month={month}")

# 2. Dropping records with Null values
df_lz_dropna = df_lz_deduped.dropna()
df_lz_dropna_count = df_lz_dropna.count()
df_rejected_dropna = df_lz_deduped.subtract(df_lz_dropna)
df_rejected_dropna.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/raw_dropped/dropped_nulls/year={year}/month={month}")

# 3. Dropping malformed locations
df_lz_valid_locations = df_lz_dropna
for column in col_locations:
    df_lz_valid_locations = df_lz_valid_locations.filter(
        (F.col(column) >= 1) & (F.col(column) <= 266)
    )
df_lz_valid_locations_count = df_lz_valid_locations.count()
df_rejected_locations = df_lz_dropna.subtract(df_lz_valid_locations)
df_rejected_locations.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/raw_dropped/dropped_locations/year={year}/month={month}")

# 4. Dropping the records with month ot year not matching with the landed partition
df_lz_dateformat = df_lz_valid_locations

df_lz_dateformat = df_lz_dateformat.withColumn("given_month", F.date_format("tpep_pickup_datetime", "MM"))\
                                    .withColumn("given_year", F.date_format("tpep_pickup_datetime", "yyyy"))
df_lz_dateformat = df_lz_dateformat.filter((F.col("given_month") == month) & (F.col("given_year") == year))
df_lz_dateformat = df_lz_dateformat.drop("given_month", "given_year")
df_lz_dateformat_count = df_lz_dateformat.count()
df_rejected_dateformat = df_lz_valid_locations.subtract(df_lz_dateformat)
df_rejected_dateformat.coalesce(1).write.mode("overwrite").parquet(f"{rejected_write_path}/raw_dropped/dropped_stray_timestamps/year={year}/month={month}")

#====================Adding Row Level Audit Fields=======================

df_final = df_lz_dateformat
df_final = df_final.withColumn(
    "ingestion_timestamp", F.current_timestamp()
).withColumn(
    "source_file_name", F.element_at(F.split(F.input_file_name(), "/"), -1)
    )

#====================Writing df_final as Parquet in Raw zone===============================

# df_final.show(20)
# print(df_final.count())
df_final.coalesce(1).write.mode("overwrite").parquet(f"{write_path}/facts/year={year}/month={month}")

#===================Job Commit==================

job.commit()
logger.info("Job 1 is completed!")