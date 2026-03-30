"""Chargement du modele et logique de prediction.

Le modele est charge UNE SEULE FOIS au demarrage du module,
puis reutilise pour toutes les requetes.
"""

import time
import numpy as np
import pandas as pd
import joblib

from app.config import (
    MODEL_PATH,
    ONNX_MODEL_PATH,
    SAMPLE_DATA_PATH,
    FEATURE_NAMES,
    THRESHOLD,
    USE_ONNX,
)
from app.logger import log_prediction

# --- Chargement au demarrage (une seule fois) ---

# Charger les donnees clients echantillon
_client_data = pd.read_csv(SAMPLE_DATA_PATH)
_client_ids = _client_data["SK_ID_CURR"].tolist()
# On garde SK_ID_CURR comme colonne (le modele a ete entraine avec)
_client_data_indexed = _client_data.set_index("SK_ID_CURR")

# Charger le modele
if USE_ONNX:
    import onnxruntime as ort

    _onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
    _model = None
    print(f"Modele ONNX charge : {ONNX_MODEL_PATH}")
else:
    _model = joblib.load(MODEL_PATH)
    _onnx_session = None
    print(f"Modele LightGBM charge : {MODEL_PATH}")

print(f"Donnees clients chargees : {len(_client_ids)} clients disponibles")


def get_client_ids() -> list[int]:
    """Retourne la liste des IDs clients disponibles."""
    return _client_ids


def predict(client_id: int) -> tuple[float, str, float]:
    """Predire la probabilite de defaut pour un client.

    Args:
        client_id: Identifiant SK_ID_CURR du client.

    Returns:
        Tuple (probabilite, decision, inference_time_ms).

    Raises:
        ValueError: Si le client_id n'existe pas dans les donnees.
    """
    if client_id not in _client_data_indexed.index:
        raise ValueError(f"Client {client_id} non trouve dans les donnees.")

    # Extraire les features du client (SK_ID_CURR inclus car le modele a ete entraine avec)
    row = _client_data[_client_data["SK_ID_CURR"] == client_id]
    features = row[FEATURE_NAMES]

    # Prediction avec mesure du temps
    start = time.perf_counter()

    if USE_ONNX and _onnx_session is not None:
        input_name = _onnx_session.get_inputs()[0].name
        result = _onnx_session.run(
            None, {input_name: features.values.astype(np.float32)}
        )
        probability = float(result[1][0][1])
    else:
        probability = float(_model.predict_proba(features)[:, 1][0])

    inference_ms = (time.perf_counter() - start) * 1000

    # Appliquer le seuil de decision
    decision = "REFUSE" if probability >= THRESHOLD else "ACCORDE"

    # Logger la prediction
    log_prediction(client_id, probability, decision, THRESHOLD, inference_ms)

    return probability, decision, inference_ms
