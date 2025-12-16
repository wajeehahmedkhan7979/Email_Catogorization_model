"""
Application configuration using Pydantic settings.

This module centralises all environment-driven configuration and provides
typed accessors with validation. It also optionally integrates with
Azure Key Vault for third‑party API secrets.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import List, Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from pydantic import BaseSettings, Field, ValidationError, validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Strongly‑typed application settings loaded from environment variables.

    Required variables:
    - AZURE_STORAGE_ACCOUNT_URL
    - INPUT_CONTAINER
    - OUTPUT_CONTAINER
    - POISON_QUEUE_NAME
    - QUEUE_NAME
    - TAXONOMY_PATH
    - INTENT_MODEL_PATH

    Optional:
    - MODEL_VERSION
    - ALLOWED_LANGUAGES
    - MAX_PAYLOAD_BYTES
    - KEY_VAULT_URL
    - THIRD_PARTY_API_SECRET_NAME
    """

    azure_storage_account_url: str = Field(
        ...,
        env="AZURE_STORAGE_ACCOUNT_URL",
        description="Azure Storage account blob/queue endpoint URL.",
    )
    input_container: str = Field(
        ...,
        env="INPUT_CONTAINER",
        description="Name of the input Blob container for raw email JSON.",
    )
    output_container: str = Field(
        ...,
        env="OUTPUT_CONTAINER",
        description="Name of the output Blob container for classified results.",
    )
    poison_queue_name: str = Field(
        ...,
        env="POISON_QUEUE_NAME",
        description="Name of the poison queue for irrecoverable messages.",
    )
    queue_name: str = Field(
        ...,
        env="QUEUE_NAME",
        description="Name of the primary input queue for email events.",
    )

    taxonomy_path: str = Field(
        ...,
        env="TAXONOMY_PATH",
        description="Filesystem path to the taxonomy JSON file.",
    )
    intent_model_path: str = Field(
        ...,
        env="INTENT_MODEL_PATH",
        description="Filesystem path to the intent classifier model pickle.",
    )
    model_version: str = Field(
        "v1.0.0",
        env="MODEL_VERSION",
        description="Logical model version identifier emitted in outputs.",
    )

    allowed_languages: List[str] = Field(
        default_factory=lambda: ["en", "es"],
        env="ALLOWED_LANGUAGES",
        description="List of ISO language codes allowed for processing.",
    )

    max_payload_bytes: int = Field(
        512_000,
        env="MAX_PAYLOAD_BYTES",
        description=(
            "Maximum allowed payload size in bytes for a single email."
        ),
    )

    # Optional secrets / Key Vault
    key_vault_url: Optional[str] = Field(
        None,
        env="KEY_VAULT_URL",
        description=(
            "Azure Key Vault URL (optional, for third‑party secrets)."
        ),
    )
    third_party_api_secret_name: Optional[str] = Field(
        None,
        env="THIRD_PARTY_API_SECRET_NAME",
        description="Key Vault secret name for any third‑party API key.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("azure_storage_account_url")
    def validate_account_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            msg = (
                "AZURE_STORAGE_ACCOUNT_URL must be a valid HTTPS URL "
                f"(got '{value}')."
            )
            raise ValueError(msg)
        return value.rstrip("/")

    @validator("allowed_languages", pre=True)
    def parse_allowed_languages(cls, value):  # type: ignore[no-untyped-def]
        """
        Allow ALLOWED_LANGUAGES as JSON or comma-separated string.
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return [lang.strip() for lang in value.split(",") if lang.strip()]
        return ["en", "es"]

    @validator("max_payload_bytes")
    def validate_max_payload_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                "MAX_PAYLOAD_BYTES must be a positive integer."
            )
        if value > 10_000_000:
            raise ValueError(
                "MAX_PAYLOAD_BYTES is unreasonably large; "
                "consider keeping it under 10MB."
            )
        return value

    def get_key_vault_secret(self) -> Optional[str]:
        """
        Resolve an optional third‑party API key from Azure Key Vault.

        Returns:
            The secret value as a string, or None if Key Vault integration is
            not configured.
        """
        if not self.key_vault_url or not self.third_party_api_secret_name:
            return None

        try:
            credential = DefaultAzureCredential()
            client = SecretClient(
                vault_url=self.key_vault_url,
                credential=credential,
            )
            secret = client.get_secret(self.third_party_api_secret_name)
            return secret.value
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read secret from Key Vault: %s", exc)
            return None


@lru_cache()
def get_settings() -> Settings:
    """
    Load and cache application settings.

    Raises:
        ValidationError: if any required configuration value is missing or invalid.
    """
    try:
        return Settings()  # type: ignore[arg-type]
    except ValidationError as exc:
        # Log a compact but informative message before bubbling up.
        logger.error("Configuration validation failed: %s", exc)
        raise
