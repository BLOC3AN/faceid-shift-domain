import os
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv
from minio import Minio

# Load environments
load_dotenv()

# Configure logger
logger = logging.getLogger("FaceClustering.MinioClient")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MinioFaceClient:
    """
    A client wrapper for MinIO to download face images.
    """
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "192.168.1.125:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "admin123")
        self.secure = os.getenv("MINIO_SECURE", "0") == "1"
        self.default_bucket = os.getenv("MINIO_BUCKET_NAME", "detected-faces")
        self.client: Optional[Minio] = None
        
        # Initialize the connection
        self.connect()

    def connect(self) -> None:
        """
        Establish connection to the MinIO server.
        """
        try:
            logger.info(f"Connecting to MinIO at {self.endpoint} (secure={self.secure})...")
            self.client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            logger.info("Successfully connected to MinIO client.")
        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}", exc_info=True)
            raise

    def parse_minio_url(self, minio_url: str) -> Tuple[str, str]:
        """
        Parses minio URL to extract bucket name and object name.
        Example: http://localhost:9000/detected-faces/source_0/face_20260605.jpg
        returns ("detected-faces", "source_0/face_20260605.jpg")
        
        Args:
            minio_url: The full URL path of the image in MinIO.
            
        Returns:
            Tuple of (bucket_name, object_name)
        """
        try:
            parsed = urlparse(minio_url)
            path = parsed.path.lstrip('/')
            parts = path.split('/')
            
            if len(parts) >= 2:
                bucket_name = parts[0]
                object_name = '/'.join(parts[1:])
                return bucket_name, object_name
            elif len(parts) == 1 and parts[0] != "":
                # Fallback to default bucket if URL path only contains filename
                return self.default_bucket, parts[0]
            else:
                raise ValueError("URL path is empty or invalid.")
        except Exception as e:
            logger.warning(f"Failed to parse MinIO URL '{minio_url}': {e}. Using fallback path extraction.")
            # Simple fallback
            filename = minio_url.split('/')[-1]
            return self.default_bucket, filename

    def download_face_image(self, minio_url: str, local_dest_path: str) -> bool:
        """
        Downloads the image specified by minio_url and saves it to local_dest_path.
        
        Args:
            minio_url: URL to the file in MinIO.
            local_dest_path: Local path where the file will be saved.
            
        Returns:
            bool: True if download succeeded, False otherwise.
        """
        if not self.client:
            raise ConnectionError("MinIO client is not connected.")
            
        bucket_name, object_name = self.parse_minio_url(minio_url)
        
        try:
            # Ensure the local directory structure exists
            local_dir = os.path.dirname(local_dest_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
                
            logger.debug(f"Downloading '{object_name}' from bucket '{bucket_name}' to '{local_dest_path}'...")
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=local_dest_path
            )
            return True
        except Exception as e:
            logger.error(f"Error downloading object '{object_name}' from bucket '{bucket_name}': {e}")
            return False

    def download_to_memory(self, minio_url: str) -> Optional[bytes]:
        """
        Downloads the image into memory buffer instead of writing to disk.
        Used for in-memory processing pipeline (normalize → write once).
        
        Args:
            minio_url: URL to the file in MinIO.
            
        Returns:
            bytes: Raw image data, or None if download failed.
        """
        if not self.client:
            raise ConnectionError("MinIO client is not connected.")
            
        bucket_name, object_name = self.parse_minio_url(minio_url)
        
        try:
            logger.debug(f"Downloading '{object_name}' from bucket '{bucket_name}' to memory...")
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            logger.error(f"Error downloading '{object_name}' to memory: {e}")
            return None
