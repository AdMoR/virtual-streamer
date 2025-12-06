from fastapi import FastAPI, HTTPException, Depends, status, Body, Query
import os
import boto3
from botocore.exceptions import ClientError
import json
from typing import Optional, List


class AsyncS3Client:
    def __init__(self, bucket_name):
        self.S3_BUCKET_NAME = bucket_name  # Use a dedicated bucket or prefix
        self.s3_client = boto3.client("s3")
        s3_resource = boto3.resource("s3")

    async def s3_put_json(self, key: str, data: dict):
        """Uploads a dictionary as a JSON object to S3."""
        self.s3_client.put_object(
            Bucket=self.S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(
                data, indent=2, default=str
            ),  # Use default=str for datetime
            ContentType="application/json",
        )

    async def s3_put_video_file(self, video_path: str, s3_prefix: str):
        """Uploads a dictionary as a JSON object to S3."""
        self.s3_client.upload_file(
            video_path,
            self.S3_BUCKET_NAME,
            f"{s3_prefix}/{os.path.basename(video_path)}",
        )

    async def s3_get_json(self, key: str) -> Optional[dict]:
        """Downloads and parses a JSON object from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.S3_BUCKET_NAME, Key=key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            print(f"Error downloading from S3 key {key}: {e}")
            raise e

    async def s3_delete_object(self, key: str):
        """Deletes an object from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.S3_BUCKET_NAME, Key=key)
        except ClientError as e:
            print(f"Error deleting S3 key {key}: {e}")
            # Decide if this should be a critical error or just logged
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"S3 delete error: {e}")

    async def s3_list_keys(self, prefix: str) -> List[str]:
        """Lists keys within a given prefix in S3."""
        keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.S3_BUCKET_NAME, Prefix=prefix):
                if "Contents" in page:
                    for item in page["Contents"]:
                        keys.append(item["Key"])
        except ClientError as e:
            print(f"Error listing S3 prefix {prefix}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"S3 list error: {e}",
            )
        return keys
