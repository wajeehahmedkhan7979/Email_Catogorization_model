import re
from typing import List, Optional

from bs4 import BeautifulSoup
from langdetect import LangDetectException, detect

from src.models import Conversation, EmailPayload

FOOTER_REGEX = re.compile(
    r"(sent from my|envoyé de mon|enviado desde mi)", re.IGNORECASE
)


def strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator="\n")


def detect_language(text: str, allowed: List[str]) -> Optional[str]:
    try:
        lang = detect(text)
        return lang if lang in allowed else None
    except LangDetectException:
        return None


def preprocess_payload(
    payload: EmailPayload, allowed_languages: List[str]
) -> Optional[Conversation]:
    if not payload.messages:
        return None

    # Choose subject from first message with content
    subject = next((m.subject or "" for m in payload.messages if m.subject), "")

    parts: List[str] = []
    attachments = []
    for message in sorted(payload.messages, key=lambda m: m.sent_at or 0):
        body = message.body or ""
        body = strip_html(body)
        body = FOOTER_REGEX.split(body)[0]
        if message.attachments:
            for att in message.attachments:
                attachments.append(att)
                parts.append(f"attachment: {att.filename}")
        parts.append(body)

    merged = "\n\n".join([p for p in parts if p])
    lang = detect_language(merged, allowed_languages)

    return Conversation(
        conversation_id=payload.conversation_id or "unknown",
        subject=subject,
        body=merged,
        language=lang,
        attachments=attachments,
        metadata={"message_count": len(payload.messages)},
    )
