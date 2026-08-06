import json
import boto3
import os
import logging
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

stepfunctions = boto3.client("stepfunctions")

STEP_FUNCTIONS_ARN = os.environ.get("STEP_FUNCTIONS_ARN", "")

def extract_partititon(s3_key):
    match = re.search(r"(\d{4})-(\d{2})", s3_key)

    if match:
        year = match.group(1)
        month = match.group(2)

    logger.info(f"year={year}, month={month}")
    return year,month

def lambda_handler(event, context):
    logger.info(f"Recieved event: {json.dumps(event)}")

    #=========Parsing the S3 event=========
    try:
        record = event["Records"][0]
        s3_key = record["s3"]["object"]["key"]
        bucket = record["s3"]["bucket"]["name"]
    except(KeyError, IndexError) as e:
        logger.error(f"Failed to parse s3 event {e}")
        raise ValueError(f"Invalid s3 event structure {e}")

    logger.info(f"File Uploaded to s3://{bucket}/{s3_key}")

    #===========Extract Partition info=========

    year, month = extract_partititon(s3_key)

    #========Building Step function input payload=========
    execution_input = {
        "bucket": bucket,
        "year": year,
        "month": month,
        "landing_path": "landing",
        "raw_path": "raw",
        "silver_path": "silver",
        "gold_path": "gold"
    }

    #=========Start step function execution==========
    execution_name = f"nyc-yellow-taxi-{year}-{month}"

    try:
        response = stepfunctions.start_execution(
            stateMachineArn = STEP_FUNCTIONS_ARN,
            name = execution_name,
            input = json.dumps(execution_input)
        )
        logger.info(f"Step Functions started: {response['executionArn']}")
    except stepfunctions.exceptions.ExecutionAlreadyExists:
        logger.warning(f"Execution {execution_name} already exists — skipping duplicate")
        return {
                    "statusCode": 200,
                    "body": f"Execution already exists for {year}-{month}"
                }
    except Exception as e:
        logger.error(f"Failed to start Step Functions: {e}")
        raise
            
    return {
            "statusCode": 200,
            "body": json.dumps({
                "message":      "Pipeline triggered successfully",
                "year":         year,
                "month":        month,
                "execution":    response["executionArn"]
            })
    }