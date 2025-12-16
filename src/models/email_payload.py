"""
Models for inbound email payloads used by the worker.

These schemas enforce strict validation of the JSON envelope received from
upstream systems before any preprocessing or classification is performed.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, root_validator, validator


class EmailPayload(BaseModel):
    """
    Inbound email JSON payload.

    This represents a single email event as produced by upstream systems,
    prior to any conversation/thread merging.
    """

    message_id: str = Field(
        ...,
        description="Unique identifier of the email.",
    )
    subject: str = Field(..., description="Subject line of the email.")
    body: str = Field(..., description="Raw body text or HTML content.")
    sender: str = Field(..., description="Sender email address.")
    recipients: List[str] = Field(
        ..., description="List of recipient email addresses."
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="List of attachment filenames associated with the email.",
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp when the email was sent/received.",
    )
    priority: Optional[str] = Field(
        None,
        description="Optional priority indicator (e.g., 'high', 'normal').",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation identifier if provided by upstream.",
    )
    thread_id: Optional[str] = Field(
        None,
        description="Thread identifier if provided instead of conversation_id.",
    )

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        validate_assignment = True

    @validator("recipients")
    def validate_recipients(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError(
                "recipients must contain at least one email address."
            )
        return value

    @validator("subject", "body", "sender")
    def non_empty_strings(cls, value: str, field):  # type: ignore[no-untyped-def]
        if not value or not value.strip():
            raise ValueError(f"{field.name} must be a non-empty string.")
        return value.strip()

    @root_validator
    def ensure_thread_identifier(cls, values):  # type: ignore[no-untyped-def]
        """
        Ensure that at least one of conversation_id or thread_id is present.
        """
        conv = values.get("conversation_id")
        thread = values.get("thread_id")
        if not conv and not thread:
            raise ValueError(
                "Either 'conversation_id' or 'thread_id' must be provided "
                "to support conversation-level merging."
            )
        return values


__all__ = ["EmailPayload", "ValidationError"]
