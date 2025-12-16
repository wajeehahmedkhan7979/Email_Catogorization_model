import json
import logging
import signal
import time
from typing import Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.queue import QueueMessage

from src.config import get_settings
from src.models import EmailPayload
from src.services.ai_service import EmbeddingService
from src.services.drift_monitor import DriftMonitor
from src.services.preprocessor import preprocess_payload
from src.services.taxonomy import Taxonomy

logger = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO)

SHOULD_STOP = False


def handle_sigterm(signum, frame):
    global SHOULD_STOP
    SHOULD_STOP = True
    logger.info("SIGTERM received, finishing current message then exiting.")


signal.signal(signal.SIGTERM, handle_sigterm)


def parse_queue_message(message: QueueMessage) -> Optional[str]:
    try:
        body = json.loads(message.content)
        return body.get("data", {}).get("url") or body.get("blob_name")
    except Exception:
        return None


def worker_loop():
    settings = get_settings()

    # Lazy imports to keep startup light for tests.
    from src.services.blob_client import BlobClientFactory
    from src.services.queue_client import QueueClientFactory

    blob_factory = BlobClientFactory(
        account_url=settings.azure_storage_account_url
    )
    queue_factory = QueueClientFactory(
        account_url=settings.azure_storage_account_url,
        queue_name=settings.queue_name,
    )
    poison_queue_factory = QueueClientFactory(
        account_url=settings.azure_storage_account_url,
        queue_name=settings.poison_queue_name,
    )
    queue = queue_factory.get_client()
    poison_queue = poison_queue_factory.get_client()
    embedder = EmbeddingService()

    # Placeholder centroid vectors; replace with persisted centroids.
    dummy_centroids = {"other": embedder.embed(["other"])[0]}
    taxonomy = Taxonomy(settings.taxonomy_path, embeddings=dummy_centroids)
    baseline = taxonomy.data.get(
        "baseline",
        {"mean_similarity": 0.8, "p10": 0.7, "p50": 0.8, "p90": 0.9},
    )
    drift_monitor = DriftMonitor(baseline=baseline, window_size=1000)

    while not SHOULD_STOP:
        messages = queue.receive_messages(
            messages_per_page=1, visibility_timeout=300
        )
        received = False
        for msg in messages:
            received = True
            blob_name = parse_queue_message(msg)
            if not blob_name:
                logger.warning(
                    "Could not parse queue message, deleting: %s", msg.id
                )
                queue.delete_message(msg)
                continue

            try:
                blob_client = blob_factory.get_client().get_blob_client(
                    container=settings.input_container, blob=blob_name
                )
                payload_text = (
                    blob_client.download_blob()
                    .readall()
                    .decode("utf-8", errors="ignore")
                )
                payload = EmailPayload.parse_raw(payload_text)
            except ResourceNotFoundError:
                logger.exception("Blob not found: %s", blob_name)
                queue.delete_message(msg)
                continue
            except Exception:
                logger.exception(
                    "Failed to parse payload for blob: %s", blob_name
                )
                queue.delete_message(msg)
                continue

            conversation = preprocess_payload(
                payload, allowed_languages=settings.allowed_languages
            )
            if not conversation:
                logger.info(
                    "Empty conversation for blob %s, deleting message.",
                    blob_name,
                )
                queue.delete_message(msg)
                continue

            embedding = embedder.embed([conversation.body])[0]
            matches = taxonomy.match_levels(embedding)
            if matches:
                drift_monitor.record(matches[0][1])
                drift_alert = drift_monitor.check()
                if drift_alert:
                    logger.warning("Taxonomy drift detected: %s", drift_alert)
            output = {
                "conversation_id": conversation.conversation_id,
                "taxonomy_version": taxonomy.version,
                "matches": [{"label": m[0], "score": m[1]} for m in matches],
                "language": conversation.language,
                "model_version": settings.model_version,
            }

            try:
                target_blob = blob_factory.get_client().get_blob_client(
                    container=settings.output_container,
                    blob=f"{conversation.conversation_id}.json",
                )
                target_blob.upload_blob(json.dumps(output), overwrite=True)
                queue.delete_message(msg)
                logger.info("Processed blob %s", blob_name)
            except Exception as exc:
                logger.exception(
                    "Processing failed for %s (attempt %d)",
                    blob_name,
                    msg.dequeue_count,
                )
                if msg.dequeue_count >= settings.max_retries:
                    try:
                        original_payload = json.loads(msg.content)
                    except Exception:
                        original_payload = {"raw": msg.content}
                    poison_payload = {
                        "original_message_id": msg.id,
                        "dequeue_count": msg.dequeue_count,
                        "failed_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "stack_hint": "worker_loop",
                        },
                        "original_payload": original_payload,
                    }
                    poison_queue.send_message(json.dumps(poison_payload))
                    queue.delete_message(msg)
                    logger.error(
                        "Message %s moved to poison queue", msg.id
                    )

        if not received:
            time.sleep(5)


if __name__ == "__main__":
    worker_loop()
