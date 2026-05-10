"""
Low-level API: Storage utilities

Presigned URL generation for MinIO objects, so browser apps can play
videos without direct MinIO access (routed through the nginx /minio/ proxy).
"""

from fastapi import APIRouter, Query
from virtual_streamer.utils.minio_client import get_storage_client

router = APIRouter()


@router.get("/storage/presign")
def presign(
    key: str = Query(..., description="Object key inside the bucket"),
    bucket: str = Query(None, description="Bucket name (defaults to configured bucket)"),
):
    """Return a presigned GET URL for a MinIO object.

    The URL uses the internal MinIO endpoint (minio:9000). Callers should
    replace that host with /minio so the nginx proxy forwards the request
    with the correct Host header, preserving the S3 V4 signature.
    """
    client = get_storage_client()
    if bucket:
        # Temporarily override bucket for cross-bucket access
        original = client.bucket
        client.bucket = bucket
        url = client.get_url(key)
        client.bucket = original
    else:
        url = client.get_url(key)
    return {"url": url}
