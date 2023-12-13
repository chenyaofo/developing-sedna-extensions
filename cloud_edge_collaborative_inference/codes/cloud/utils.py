import boto3
import tempfile
import os
import logging

LOG = logging.getLogger(__name__)
access_key = "EGKIKHO2SD0X2ILV4K67"
secret_key = "UDkZAbj0gZXIKGRCAjYMav8bqQr4zmOXSa02SRDF"

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

    s3 = boto3.client(
        service_name='s3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url='https://ceph-s3-b7-1.scut-smil.cn',
        verify=True
    )
    print(bucket_name,s3_file_name)
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3.download_file(bucket_name, s3_file_name, temp_file.name)
            LOG.info(
                f"Downloaded {s3_file_name} to temporary file {temp_file.name}")
            return temp_file.name
    except Exception as e:
        LOG.error(f"Error: {e}")
        return None
