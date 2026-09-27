import os

import boto3
from dotenv import load_dotenv

load_dotenv()


class B2Storage:
    def __init__(self):
        endpoint = os.getenv("B2_ENDPOINT")
        access_key = os.getenv("B2_ACCESS_KEY_ID")
        secret_key = os.getenv("B2_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("B2_BUCKET_NAME")

        self.bucket_name = bucket_name

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-005",
        )


    def upload_pdf(self, pdf_path, paper_id):
        object_key = f"raw/{paper_id}.pdf"

        self.client.upload_file(
            pdf_path,
            self.bucket_name,
            object_key,
        )

        return object_key


    
    def pdf_exists(self, paper_id):
        object_key = f"raw/{paper_id}.pdf"

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_key,
            )
            return True
        except self.client.exceptions.ClientError:
            return False


    def download_pdf(self, paper_id, output_path):
        object_key = f"raw/{paper_id}.pdf"

        self.client.download_file(
            self.bucket_name,
            object_key,
            output_path,
        )

if __name__ == "__main__":
    storage = B2Storage()
    response = storage.client.list_objects_v2(Bucket=storage.bucket_name)

    print("Connected to B2!")
    print("Bucket:", storage.bucket_name)

# if __name__ == "__main__":
#     storage = B2Storage()

#     pdf_path = "data/2608.20316v1_Pandora's AI Model Routing Box Efficient Allocation with Costly Value Estimation.pdf"
#     paper_id = "2608.20316v1_Pandora's AI Model Routing Box Efficient Allocation with Costly Value Estimation"

#     key = storage.upload_pdf(pdf_path, paper_id)

#     print("Uploaded:", key)
#     print("Exists:", storage.pdf_exists(paper_id))

#     storage.download_pdf(paper_id, "downloaded_test.pdf")
#     print("Downloaded!")