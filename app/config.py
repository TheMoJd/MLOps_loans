"""Configuration de l'application de scoring credit."""

import os
import json

# Chemins relatifs depuis la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "model", "model_optimized.onnx")
CONFIG_PATH = os.path.join(BASE_DIR, "model", "config.json")
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_clients.csv")
REFERENCE_DATA_PATH = os.path.join(BASE_DIR, "data", "reference_data.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")

# Charger la config du modele
with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

THRESHOLD = MODEL_CONFIG["threshold"]
FEATURE_NAMES = MODEL_CONFIG["feature_names"]
N_FEATURES = MODEL_CONFIG["n_features"]

# Mode ONNX (active via variable d'environnement)
USE_ONNX = os.getenv("USE_ONNX", "false").lower() == "true"

# Port de l'API
PORT = int(os.getenv("PORT", 7860))
