from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class Attachment(BaseModel):
    filename: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = Field(None, ge=0)
    download_url: Optional[HttpUrl] = None


class EmailMessage(BaseModel):
    message_id: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    sent_at: Optional[datetime] = None
    in_reply_to: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    attachments: List[Attachment] = Field(default_factory=list)

    @validator("body")
    def normalize_body(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.replace("\x00", "").strip() or None


class EmailPayload(BaseModel):
    conversation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    messages: List[EmailMessage]


class Conversation(BaseModel):
    conversation_id: str
    subject: str
    body: str
    language: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
