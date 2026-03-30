"""Tests unitaires pour le modele de scoring."""

import numpy as np
import pandas as pd
import pytest

from app.config import THRESHOLD, FEATURE_NAMES


class TestModelPredictions:
    """Tests sur les predictions du modele LightGBM."""

    def test_predict_proba_returns_valid_range(self, model, sample_features):
        """La probabilite doit etre entre 0 et 1."""
        proba = model.predict_proba(sample_features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_predict_proba_shape(self, model, sample_features):
        """predict_proba retourne 2 colonnes (classe 0 et classe 1)."""
        proba = model.predict_proba(sample_features)
        assert proba.shape == (1, 2)

    def test_probabilities_sum_to_one(self, model, sample_features):
        """Les probabilites des deux classes doivent sommer a 1."""
        proba = model.predict_proba(sample_features)
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_batch_prediction(self, model, sample_data):
        """Le modele doit gerer les predictions par lot."""
        features = sample_data[FEATURE_NAMES].head(10)
        proba = model.predict_proba(features)[:, 1]
        assert len(proba) == 10
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_handles_nan_values(self, model, sample_features):
        """LightGBM gere nativement les NaN — pas d'erreur attendue."""
        features_with_nan = sample_features.copy()
        features_with_nan.iloc[0, 0] = np.nan
        features_with_nan.iloc[0, 5] = np.nan
        proba = model.predict_proba(features_with_nan)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_extreme_values(self, model, sample_features):
        """Le modele doit retourner une proba valide meme avec des valeurs extremes."""
        features_extreme = sample_features.copy()
        features_extreme.iloc[0, 0] = 1e10  # valeur tres grande
        features_extreme.iloc[0, 1] = -1e10  # valeur tres negative
        proba = model.predict_proba(features_extreme)[:, 1]
        assert 0.0 <= proba[0] <= 1.0


class TestThresholdDecision:
    """Tests sur la logique de decision avec le seuil."""

    def test_threshold_is_valid(self):
        """Le seuil doit etre entre 0 et 1."""
        assert 0.0 < THRESHOLD < 1.0

    def test_high_proba_means_refused(self):
        """Probabilite >= seuil => credit REFUSE."""
        proba = THRESHOLD + 0.1
        decision = "REFUSE" if proba >= THRESHOLD else "ACCORDE"
        assert decision == "REFUSE"

    def test_low_proba_means_accepted(self):
        """Probabilite < seuil => credit ACCORDE."""
        proba = THRESHOLD - 0.1
        decision = "REFUSE" if proba >= THRESHOLD else "ACCORDE"
        assert decision == "ACCORDE"

    def test_exact_threshold_means_refused(self):
        """Probabilite == seuil => credit REFUSE (convention >=)."""
        proba = THRESHOLD
        decision = "REFUSE" if proba >= THRESHOLD else "ACCORDE"
        assert decision == "REFUSE"


class TestFeatureConsistency:
    """Tests sur la coherence des features."""

    def test_feature_count(self, sample_features):
        """Le nombre de features doit correspondre a la config."""
        assert sample_features.shape[1] == len(FEATURE_NAMES)

    def test_feature_names_match(self, sample_features):
        """Les noms de colonnes doivent correspondre a la config."""
        assert list(sample_features.columns) == FEATURE_NAMES

    def test_no_target_in_features(self):
        """TARGET ne doit pas etre dans les features."""
        assert "TARGET" not in FEATURE_NAMES
