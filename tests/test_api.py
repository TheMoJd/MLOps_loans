"""Tests d'integration pour l'API de scoring."""

import pytest
from app.model import predict, get_client_ids


class TestPredictFunction:
    """Tests sur la fonction predict() utilisee par l'API."""

    def test_predict_returns_tuple(self, valid_client_id):
        """predict() retourne un tuple (proba, decision, time)."""
        result = predict(valid_client_id)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_predict_probability_type(self, valid_client_id):
        """La probabilite est un float."""
        proba, _, _ = predict(valid_client_id)
        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0

    def test_predict_decision_values(self, valid_client_id):
        """La decision est ACCORDE ou REFUSE."""
        _, decision, _ = predict(valid_client_id)
        assert decision in ("ACCORDE", "REFUSE")

    def test_predict_inference_time_positive(self, valid_client_id):
        """Le temps d'inference est positif."""
        _, _, inference_ms = predict(valid_client_id)
        assert inference_ms > 0

    def test_predict_invalid_client_raises(self):
        """Un client inexistant leve une ValueError."""
        with pytest.raises(ValueError, match="non trouve"):
            predict(999999999)

    def test_predict_multiple_clients(self):
        """Plusieurs predictions consecutives fonctionnent."""
        client_ids = get_client_ids()[:5]
        for cid in client_ids:
            proba, decision, _ = predict(cid)
            assert 0.0 <= proba <= 1.0
            assert decision in ("ACCORDE", "REFUSE")


class TestGetClientIds:
    """Tests sur la fonction get_client_ids()."""

    def test_returns_list(self):
        """Retourne une liste."""
        ids = get_client_ids()
        assert isinstance(ids, list)

    def test_not_empty(self):
        """La liste n'est pas vide."""
        ids = get_client_ids()
        assert len(ids) > 0

    def test_ids_are_integers(self):
        """Les IDs sont des entiers."""
        ids = get_client_ids()
        assert all(isinstance(i, int) for i in ids)


class TestInvalidInputTypes:
    """Tests sur le rejet des types incorrects en entree de l'API."""

    def test_predict_string_client_id_raises(self):
        """Un client_id de type str (ex: 'abc') doit lever une ValueError."""
        with pytest.raises(ValueError, match="non trouve"):
            predict("abc")

    def test_predict_none_client_id_raises(self):
        """Un client_id None doit lever une ValueError."""
        with pytest.raises(ValueError, match="non trouve"):
            predict(None)

    def test_predict_negative_client_id_raises(self):
        """Un client_id negatif (impossible metier) doit lever une ValueError."""
        with pytest.raises(ValueError, match="non trouve"):
            predict(-1)

    def test_predict_zero_client_id_raises(self):
        """Un client_id egal a 0 (non reference) doit lever une ValueError."""
        with pytest.raises(ValueError, match="non trouve"):
            predict(0)

    def test_predict_float_client_id_raises(self):
        """Un client_id en float non entier doit lever une ValueError."""
        with pytest.raises(ValueError, match="non trouve"):
            predict(12345.67)

    def test_predict_list_client_id_raises(self):
        """Un client_id de type list doit lever une erreur (pas un scalaire hashable)."""
        with pytest.raises((ValueError, TypeError)):
            predict([123456])
