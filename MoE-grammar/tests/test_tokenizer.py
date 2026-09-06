import numpy as np

from moe_grammar.tokenizer import GMMTokenizer


def _two_cluster_training(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=-4.0, scale=0.3, size=(400, 3))
    right = rng.normal(loc=4.0, scale=0.3, size=(400, 3))
    return np.vstack([left, right]).astype(np.float32)


def test_unknown_posterior_flags_off_manifold_queries() -> None:
    tokenizer = GMMTokenizer(n_words=2, seed=5, n_init=1).fit(_two_cluster_training())
    inside = np.asarray([[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    far = np.asarray([[60.0, -60.0, 60.0]], dtype=np.float32)
    _, _, inside_lexical = tokenizer.transform(inside)
    _, _, far_lexical = tokenizer.transform(far)
    inside_posterior = tokenizer.unknown_posterior(inside_lexical)
    far_posterior = tokenizer.unknown_posterior(far_lexical)
    assert np.all(inside_posterior < 0.01)
    assert far_posterior[0] > 0.99
    assert np.all((inside_posterior >= 0.0) & (inside_posterior <= 1.0))


def test_unknown_posterior_requires_fitting() -> None:
    tokenizer = GMMTokenizer(n_words=2)
    try:
        tokenizer.unknown_posterior(np.zeros(3))
    except ValueError as error:
        assert "fitted" in str(error)
    else:  # pragma: no cover - guards against silently scoring an unfitted model
        raise AssertionError("expected unfitted tokenizer to raise")


def test_hard_assignment_still_covers_every_word() -> None:
    values = _two_cluster_training()
    tokenizer = GMMTokenizer(n_words=2, seed=5, n_init=1).fit(values)
    words, component, lexical = tokenizer.transform(values)
    assert set(np.unique(words).tolist()) == {0, 1}
    assert component.shape == (len(values), 2)
    assert np.isfinite(lexical).all()
