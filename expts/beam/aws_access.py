import os
import boto3

# https://docs.beam.cloud/examples/s3-schedule

class Boto3Client:
    def __init__(self):
        self.boto3_client = boto3.session.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )

    def download_from_s3(self, bucket_name, download_path):
        s3_client = self.boto3_client.resource("s3").Bucket(bucket_name)

        for s3_object in s3_client.objects.all():
            filename = os.path.split(s3_object.key)
            s3_client.download_file(s3_object.key, f"{download_path}/{filename}")

    def upload_to_s3(self, bucket_name, file_body, key):
        s3_client = self.boto3_client.resource("s3").Bucket(bucket_name)
        s3_client.put_object(Body=file_body, Key=key)
