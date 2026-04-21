from minio import Minio
from minio.error import S3Error
import io
from pathlib import Path


class ImageStorage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.client = Minio(
            endpoint,  # e.g. "localhost:9000"
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # 本地开发用False，生产用True
        )
        self.bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            # 设置公开读策略（按需，也可以用presigned url）
            policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self.bucket}/*"]
                }]
            }
            import json
            self.client.set_bucket_policy(self.bucket, json.dumps(policy))

    def upload_image(self, image_path: str, object_name: str) -> str:
        """上传图片，返回访问URL"""
        with open(image_path, "rb") as f:
            data = f.read()

        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="image/png"
        )
        # 返回访问URL
        return f"http://{self.client._base_url.host}/{self.bucket}/{object_name}"

    def upload_pil_image(self, img, object_name: str) -> str:
        """直接上传PIL Image对象，不落盘"""
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        data = buf.getvalue()

        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="image/png"
        )
        return f"http://{self.client._base_url.host}/{self.bucket}/{object_name}"

    def get_presigned_url(self, object_name: str, expires_hours: int = 24) -> str:
        """生成临时访问URL（私有bucket用）"""
        from datetime import timedelta
        return self.client.presigned_get_object(
            self.bucket, object_name,
            expires=timedelta(hours=expires_hours)
        )