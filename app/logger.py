"""Logging structure JSON pour les predictions en production."""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

from app.config import LOG_DIR, LOG_PATH

# Creer le repertoire de logs s'il n'existe pas
os.makedirs(LOG_DIR, exist_ok=True)

# Logger dedie aux predictions (format JSONL)
prediction_logger = logging.getLogger("prediction_logger")
prediction_logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3  # 10 MB max
)
handler.setFormatter(logging.Formatter("%(message)s"))
prediction_logger.addHandler(handler)


def log_prediction(
    client_id: int,
    probability: float,
    decision: str,
    threshold: float,
    inference_time_ms: float,
) -> None:
    """Enregistre une prediction dans le fichier de logs JSONL."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_id": int(client_id),
        "probability": round(float(probability), 6),
        "decision": decision,
        "threshold": threshold,
        "inference_time_ms": round(inference_time_ms, 2),
    }
    prediction_logger.info(json.dumps(entry))
