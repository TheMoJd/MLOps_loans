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


class TestBusinessRangeValues:
    """Tests sur les valeurs hors plages metier (revenu nul, age negatif, etc.).

    Le modele doit rester robuste : retourner une probabilite valide [0, 1]
    meme si une feature contient une valeur absurde d'un point de vue metier.
    Cela permet de detecter les regressions de robustesse.
    """

    def test_zero_income(self, model, sample_features):
        """Revenu total = 0 (impossible metier) : proba doit rester dans [0, 1]."""
        features = sample_features.copy()
        features.loc[:, "AMT_INCOME_TOTAL"] = 0
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_negative_income(self, model, sample_features):
        """Revenu negatif (impossible metier) : pas de crash."""
        features = sample_features.copy()
        features.loc[:, "AMT_INCOME_TOTAL"] = -50000
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_zero_credit_amount(self, model, sample_features):
        """Montant de credit demande = 0 : pas de crash."""
        features = sample_features.copy()
        features.loc[:, "AMT_CREDIT"] = 0
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_positive_days_birth_invalid_age(self, model, sample_features):
        """DAYS_BIRTH positif (= ne dans le futur, impossible metier).

        Dans ce dataset, DAYS_BIRTH est toujours negatif (jours avant la demande).
        Un age negatif revient a mettre DAYS_BIRTH > 0.
        """
        features = sample_features.copy()
        features.loc[:, "DAYS_BIRTH"] = 5  # "ne il y a -5 jours"
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_negative_children_count(self, model, sample_features):
        """Nombre d'enfants negatif (impossible metier) : pas de crash."""
        features = sample_features.copy()
        features.loc[:, "CNT_CHILDREN"] = -3
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_unrealistic_children_count(self, model, sample_features):
        """Nombre d'enfants absurde (100) : pas de crash."""
        features = sample_features.copy()
        features.loc[:, "CNT_CHILDREN"] = 100
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0

    def test_all_zero_features(self, model, sample_features):
        """Toutes les features numeriques a 0 : pas de crash."""
        features = sample_features.copy()
        numeric_cols = features.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            features[col] = 0
        proba = model.predict_proba(features)[:, 1]
        assert 0.0 <= proba[0] <= 1.0


class TestInputShapeValidation:
    """Tests sur la validation de la forme des donnees d'entree."""

    def test_missing_column_raises(self, model, sample_features):
        """Une colonne manquante doit lever une erreur du modele."""
        import pandas as pd

        features_missing = sample_features.drop(columns=["AMT_INCOME_TOTAL"])
        with pytest.raises((ValueError, KeyError, Exception)):
            model.predict_proba(features_missing)

    def test_wrong_feature_count_raises(self, model, sample_features):
        """Un nombre incorrect de colonnes doit lever une erreur."""
        features_truncated = sample_features.iloc[:, :10]
        with pytest.raises((ValueError, Exception)):
            model.predict_proba(features_truncated)

    def test_empty_dataframe_raises(self, model):
        """Un DataFrame vide doit lever une erreur."""
        import pandas as pd

        empty_df = pd.DataFrame(columns=FEATURE_NAMES)
        with pytest.raises((ValueError, Exception)):
            model.predict_proba(empty_df)

    def test_string_in_numeric_column_raises(self, model, sample_features):
        """Du texte dans une colonne numerique doit lever une erreur."""
        features = sample_features.copy().astype(object)
        features.loc[:, "AMT_INCOME_TOTAL"] = "pas un nombre"
        with pytest.raises((ValueError, TypeError, Exception)):
            model.predict_proba(features)
