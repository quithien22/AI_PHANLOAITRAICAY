import numpy as np
import pytest

from src.experiments import EXPERIMENTS
from src.train import ALGORITHMS, build_model, fit_model


def _toy_data(n=120, n_features=10, seed=0):
    """Hai cụm dữ liệu tách được, đủ để mọi thuật toán học được."""
    rng = np.random.RandomState(seed)
    X0 = rng.normal(loc=-2.0, size=(n // 2, n_features))
    X1 = rng.normal(loc=2.0, size=(n // 2, n_features))
    return np.vstack([X0, X1]), np.array([0] * (n // 2) + [1] * (n // 2))


@pytest.mark.parametrize("cfg", EXPERIMENTS, ids=[c["name"] for c in EXPERIMENTS])
def test_every_experiment_trains_and_predicts(cfg):
    X, y = _toy_data()
    model, metrics = fit_model(cfg, X, y, X, y)
    assert metrics["val_acc"] > 0.9
    assert model.predict_proba(X[:3]).shape == (3, 2)


def test_all_algorithms_are_used_in_experiments():
    assert {c["algo"] for c in EXPERIMENTS} == set(ALGORITHMS)


def test_build_model_rejects_unknown_algo():
    with pytest.raises(ValueError):
        build_model({"name": "x", "algo": "khong_ton_tai"})
