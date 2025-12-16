import pytest

from src.models.email_payload import EmailPayload
from src.services.preprocessor import Conversation, preprocess_payload


@pytest.fixture
def base_payload() -> EmailPayload:
    return EmailPayload(
        message_id="m-1",
        subject="Test subject",
        body="Hello, this is a body.",
        sender="user@example.com",
        recipients=["other@example.com"],
        attachments=[],
        timestamp="2024-01-01T12:00:00Z",
        conversation_id="c-1",
    )


def test_non_allowed_language_filtered(base_payload: EmailPayload) -> None:
    payload = base_payload.copy(
        update={"body": "これは日本語のメールです。"}, deep=True
    )
    conv = preprocess_payload(payload, allowed_languages=["en"])
    assert conv is None


def test_spam_filtering() -> None:
    payload = EmailPayload(
        message_id="m-2",
        subject="Free trial just for you",
        body="Click here to unsubscribe and buy now!",
        sender="spam@example.com",
        recipients=["user@example.com"],
        attachments=[],
        timestamp="2024-01-01T12:00:00Z",
        conversation_id="c-2",
    )
    conv = preprocess_payload(payload, allowed_languages=["en"])
    assert conv is None


def test_signature_footer_removed(base_payload: EmailPayload) -> None:
    body = "Real content.\n\nSent from my iPhone"
    payload = base_payload.copy(update={"body": body}, deep=True)
    conv = preprocess_payload(payload, allowed_languages=["en"])
    assert isinstance(conv, Conversation)
    assert "Sent from my iPhone" not in conv.body
    assert "Real content." in conv.body


def test_conversation_merging_basic(base_payload: EmailPayload) -> None:
    conv = preprocess_payload(base_payload, allowed_languages=["en"])
    assert isinstance(conv, Conversation)
    assert conv.conversation_id == "c-1"
    assert "Test subject" in conv.body
    assert "Hello, this is a body." in conv.body
    assert 0.0 <= conv.thread_consistency <= 1.0

