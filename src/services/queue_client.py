from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient


class QueueClientFactory:
    def __init__(self, account_url: str, queue_name: str, credential: Optional[DefaultAzureCredential] = None):
        self.account_url = account_url
        self.queue_name = queue_name
        self.credential = credential or DefaultAzureCredential()
        self._client: Optional[QueueClient] = None

    def get_client(self) -> QueueClient:
        if self._client is None:
            self._client = QueueClient(
                account_url=self.account_url,
                queue_name=self.queue_name,
                credential=self.credential,
            )
        return self._client

