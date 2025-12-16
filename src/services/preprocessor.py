"""
Preprocessing logic for inbound email payloads.

This module provides a `preprocess_payload` function that validates language,
filters spam, strips signatures/footers, and merges threaded messages into a
single Conversation object suitable for embedding and classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

from src.models.email_payload import EmailPayload

# Make language detection deterministic across runs.
DetectorFactory.seed = 42

FOOTER_REGEX = re.compile(
    r"(sent from my (iphone|android)|envoyé de mon|enviado desde mi)",
    re.IGNORECASE,
)


SPAM_KEYWORDS = [
    "unsubscribe",
    "free trial",
    "viagra",
    "buy now",
]


@dataclass
class Conversation:
    """
    Cleaned, merged conversation ready for embeddings and classification.

    Attributes:
        conversation_id: Stable identifier for the conversation/thread.
        body: Cleaned multi-message text content.
        language: Detected ISO language code.
        thread_consistency: Heuristic in [0, 1] estimating how coherent the
            thread is (1.0 = highly consistent).
    """

    conversation_id: str
    body: str
    language: str
    thread_consistency: float


def _strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator="\n")


def _detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None


def _is_spam(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SPAM_KEYWORDS)


def _strip_footer(text: str) -> str:
    match = FOOTER_REGEX.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def _estimate_thread_consistency(parts: List[str]) -> float:
    """
    Rough heuristic: more similar-length and non-empty segments indicate
    a more coherent conversation.
    """
    non_empty = [p for p in parts if p.strip()]
    if not non_empty:
        return 0.0
    lengths = [len(part) for part in non_empty]
    avg = sum(lengths) / len(lengths)
    variance = sum((length - avg) ** 2 for length in lengths) / len(lengths)
    # Normalise: smaller variance -> higher consistency.
    score = max(0.0, min(1.0, 1.0 / (1.0 + variance / (avg + 1.0))))
    return float(score)


def preprocess_payload(
    payload: EmailPayload,
    allowed_languages: List[str],
) -> Optional[Conversation]:
    """
    Preprocess a single email payload and derive a Conversation.

    Steps:
    1. Decode/normalise body and subject; immediately reject empty content.
    2. Strip HTML and signatures/footers.
    3. Detect language and reject non-allowed languages.
    4. Perform simple spam filtering via keyword lists.
    5. Build a Conversation with a heuristic thread_consistency score.
    """
    subject = (payload.subject or "").strip()
    body = (payload.body or "").strip()
    if not subject and not body:
        return None

    # Strip HTML and signatures.
    clean_body = _strip_html(body)
    clean_body = _strip_footer(clean_body)

    merged_text = f"{subject}\n\n{clean_body}".strip()
    if not merged_text:
        return None

    # Language detection and filtering.
    lang = _detect_language(merged_text)
    if not lang or lang not in allowed_languages:
        return None

    # Spam filtering.
    if _is_spam(merged_text):
        return None

    # Thread consistency: for a single-message payload treat as trivial thread.
    thread_consistency = _estimate_thread_consistency([merged_text])

    conv_id = (
        payload.conversation_id
        or payload.thread_id
        or payload.message_id
    )

    return Conversation(
        conversation_id=conv_id,
        body=merged_text,
        language=lang,
        thread_consistency=thread_consistency,
    )
