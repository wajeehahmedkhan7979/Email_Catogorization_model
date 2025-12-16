"""
Helpers for interacting with Azure Storage Queues.

Provides a small factory for creating queue clients and sending JSON messages.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient, QueueServiceClient


class QueueClientFactory:
    """
    Factory for Azure Storage Queue clients.

    This uses DefaultAzureCredential so it works in local dev (with Azure CLI)
    and in production (Managed Identity).
    """

    def __init__(
        self,
        account_url: str,
        queue_name: str,
        credential: Optional[DefaultAzureCredential] = None,
    ) -> None:
        self.account_url = account_url
        self.queue_name = queue_name
        self.credential = credential or DefaultAzureCredential()
        self._service_client: Optional[QueueServiceClient] = None
        self._queue_client: Optional[QueueClient] = None

    def _get_service_client(self) -> QueueServiceClient:
        if self._service_client is None:
            self._service_client = QueueServiceClient(
                account_url=self.account_url,
                credential=self.credential,
            )
        return self._service_client

    def get_client(self) -> QueueClient:
        """
        Get (and lazily create) a client for the configured queue.
        """
        if self._queue_client is None:
            self._queue_client = self._get_service_client().get_queue_client(
                self.queue_name
            )
        return self._queue_client

    def send_json_message(self, payload: Dict[str, Any]) -> None:
        """
        Serialize a dictionary as JSON and send it as a queue message.
        """
        body = json.dumps(payload, ensure_ascii=False)
        self.get_client().send_message(body)
