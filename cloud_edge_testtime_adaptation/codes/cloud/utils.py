import boto3
import tempfile
import os
import logging

LOG = logging.getLogger(__name__)


def download_file_to_temp(s3_path):
    """
    Download a file from an S3 path to a temporary file.

    :param s3_path: S3 path of the file (e.g., s3://bucket-name/path/to/file).
    :return: Path to the temporary file.
    """
    # Remove 's3://' prefix
    s3_path = s3_path[5:]

    # Find the first '/' to split bucket name and file name
    first_slash_index = s3_path.find('/')
    bucket_name = s3_path[:first_slash_index]
    s3_file_name = s3_path[first_slash_index + 1:]

    s3 = boto3.client('s3')
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3.download_file(bucket_name, s3_file_name, temp_file.name)
            LOG.info(
                f"Downloaded {s3_file_name} to temporary file {temp_file.name}")
            return temp_file.name
    except Exception as e:
        LOG.error(f"Error: {e}")
        return None

def list_images_in_s3_path(s3_path):
    """
    List all images in a specified S3 path with full S3 URI.

    :param s3_path: Full S3 path (e.g., s3://bucket-name/path/).
    :return: List of full S3 URIs for the image files.
    """
    # Remove 's3://' prefix and split to get the bucket name and prefix
    bucket_name, prefix = s3_path[5:].split('/', 1)
    s3 = boto3.client('s3')
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []

    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if any(obj['Key'].lower().endswith(ext) for ext in image_extensions):
                    full_path = f"s3://{bucket_name}/{obj['Key']}"
                    image_paths.append(full_path)
    except Exception as e:
        print(f"Error: {e}")

    return image_paths