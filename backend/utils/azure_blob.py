import os
import logging
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from constants import USERS_FOLDER

class AzureBlobHelper:
    """
    Helper class for Azure Blob Storage operations.
    Maintains a fallback if AZURE_STORAGE_CONNECTION_STRING is not set.
    """
    def __init__(self):
        self.conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_CONTAINER_NAME", "smartml-storage")
        self.enabled = bool(self.conn_str)
        
        if self.enabled:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.conn_str)
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                # Auto-create container if it doesn't exist
                if not self.container_client.exists():
                    self.container_client.create_container()
                logging.info("[AZURE] Azure Blob Storage initialized successfully.")
            except Exception as e:
                logging.error(f"[AZURE] Failed to initialize Azure Blob Storage client: {e}")
                self.enabled = False

    def upload_file(self, local_path: Path, blob_name: str) -> bool:
        if not self.enabled:
            return False
        try:
            blob_name = blob_name.replace("\\", "/")
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            logging.info(f"[AZURE] Uploaded {local_path} to blob {blob_name}")
            return True
        except Exception as e:
            logging.error(f"[AZURE] Failed to upload file {local_path}: {e}")
            return False

    def download_file(self, blob_name: str, local_path: Path) -> bool:
        if not self.enabled:
            return False
        try:
            blob_name = blob_name.replace("\\", "/")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logging.info(f"[AZURE] Downloaded blob {blob_name} to {local_path}")
            return True
        except Exception as e:
            logging.error(f"[AZURE] Failed to download blob {blob_name}: {e}")
            return False

    def download_blob_to_memory(self, blob_name: str) -> str:
        """Download blob content directly into a string in memory (no disk I/O)."""
        if not self.enabled:
            return ""
        try:
            blob_name = blob_name.replace("\\", "/")
            blob_client = self.container_client.get_blob_client(blob_name)
            if not blob_client.exists():
                return ""
            return blob_client.download_blob().readall().decode("utf-8")
        except Exception as e:
            logging.error(f"[AZURE] Failed to download blob to memory {blob_name}: {e}")
            return ""

    def get_blob_stream(self, blob_name: str):
        """Returns a generator to stream blob chunks directly for FastAPI StreamingResponse."""
        if not self.enabled:
            raise ValueError("Azure Blob Storage is not enabled")
        blob_name = blob_name.replace("\\", "/")
        blob_client = self.container_client.get_blob_client(blob_name)
        if not blob_client.exists():
            raise FileNotFoundError(f"Blob {blob_name} not found")
        
        def chunk_generator():
            stream = blob_client.download_blob()
            for chunk in stream.chunks():
                yield chunk
        return chunk_generator()

    def delete_file(self, blob_name: str) -> bool:
        if not self.enabled:
            return False
        try:
            blob_name = blob_name.replace("\\", "/")
            blob_client = self.container_client.get_blob_client(blob_name)
            if blob_client.exists():
                blob_client.delete_blob()
                logging.info(f"[AZURE] Deleted blob {blob_name}")
                return True
            return False
        except Exception as e:
            logging.error(f"[AZURE] Failed to delete blob {blob_name}: {e}")
            return False

    def list_files(self, prefix: str) -> list:
        if not self.enabled:
            return []
        try:
            prefix = prefix.replace("\\", "/")
            blob_list = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blob_list]
        except Exception as e:
            logging.error(f"[AZURE] Failed to list blobs with prefix {prefix}: {e}")
            return []

azure_blob_helper = AzureBlobHelper()
