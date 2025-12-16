from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


class BlobClientFactory:
    def __init__(self, account_url: str, credential: Optional[DefaultAzureCredential] = None):
        self.account_url = account_url
        self.credential = credential or DefaultAzureCredential()
        self._client: Optional[BlobServiceClient] = None

    def get_client(self) -> BlobServiceClient:
        if self._client is None:
            self._client = BlobServiceClient(account_url=self.account_url, credential=self.credential)
        return self._client

