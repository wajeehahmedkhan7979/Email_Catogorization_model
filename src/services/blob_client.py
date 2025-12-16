"""
Helpers for interacting with Azure Blob Storage.

This module exposes a small factory around BlobServiceClient that provides
typed helpers for JSON upload with a simple "upload then promote" pattern
to reduce the chance of readers observing partially written blobs.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, BlobServiceClient

logger = logging.getLogger(__name__)


class BlobClientFactory:
    """
    Factory for Azure Blob clients using DefaultAzureCredential.

    Usage:
        factory = BlobClientFactory(account_url)
        client = factory.get_blob_client("container", "name.json")
    """

    def __init__(
        self,
        account_url: str,
        credential: Optional[DefaultAzureCredential] = None,
    ) -> None:
        self.account_url = account_url
        self.credential = credential or DefaultAzureCredential()
        self._client: Optional[BlobServiceClient] = None

    def _get_service_client(self) -> BlobServiceClient:
        if self._client is None:
            self._client = BlobServiceClient(
                account_url=self.account_url,
                credential=self.credential,
            )
        return self._client

    def get_blob_client(self, container: str, blob: str) -> BlobClient:
        """
        Get a low-level BlobClient for a specific container/blob.
        """
        return self._get_service_client().get_blob_client(
            container=container,
            blob=blob,
        )

    def upload_json(
        self,
        container: str,
        blob: str,
        data: Dict[str, Any],
        overwrite: bool = True,
    ) -> None:
        """
        Upload JSON data to Blob Storage using a temp blob then copy+promote.

        Azure Blob Storage does not support true atomic renames; this pattern
        minimises the window where readers might see partially written content:

        1. Upload JSON to a temporary blob.
        2. Copy from temp blob to the target blob.
        3. Delete the temporary blob.
        """
        service = self._get_service_client()
        temp_name = f"{blob}.tmp-{uuid.uuid4().hex}"
        temp_client = service.get_blob_client(
            container=container,
            blob=temp_name,
        )
        target_client = service.get_blob_client(
            container=container,
            blob=blob,
        )

        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

        try:
            temp_client.upload_blob(payload, overwrite=True)
            copy_props = target_client.start_copy_from_url(temp_client.url)
            logger.debug(
                "Started copy from temp blob %s to %s, status=%s",
                temp_name,
                blob,
                getattr(copy_props, "copy_status", "unknown"),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to upload JSON blob %s: %s",
                blob,
                exc,
            )
            raise
        finally:
            try:
                temp_client.delete_blob()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning(
                    "Failed to delete temp blob %s: %s",
                    temp_name,
                    exc,
                )
