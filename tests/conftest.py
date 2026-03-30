"""Fixtures pytest pour les tests de l'API de scoring."""

import pytest
import pandas as pd
import joblib

from app.config import MODEL_PATH, SAMPLE_DATA_PATH, FEATURE_NAMES


@pytest.fixture(scope="session")
def model():
    """Charge le modele une seule fois pour toute la session de tests."""
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def sample_data():
    """Charge les donnees clients echantillon."""
    df = pd.read_csv(SAMPLE_DATA_PATH)
    return df


@pytest.fixture(scope="session")
def valid_client_id(sample_data):
    """Retourne un ID client valide."""
    return int(sample_data["SK_ID_CURR"].iloc[0])


@pytest.fixture(scope="session")
def sample_features(sample_data):
    """Retourne les features d'un client sous forme de DataFrame (SK_ID_CURR inclus)."""
    row = sample_data.iloc[[0]]
    return row[FEATURE_NAMES]  # FEATURE_NAMES inclut SK_ID_CURR
