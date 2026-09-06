import numpy as np

from moe_grammar.discrete_phenotype import chord_codes
from moe_grammar.flow_word import (
    FlowChordTokenizer,
    FlowTrackProjector,
    PositionFactorialContextGrammar,
    flow_motif_indicators,
    flow_query_words,
)


def test_flow_projector_preserves_semantic_directions() -> None:
    projector = FlowTrackProjector()
    projector.center_ = np.zeros(10)
    projector.scale_ = np.ones(10)
    primitives = np.zeros((2, 10, 10), dtype=np.float32)
    primitives[1, :, 0] = 4.0
    primitives[1, :, 4:6] = 2.0
    primitives[1, :, 6] = -3.0
    primitives[1, :, 7:10] = 3.0
    tracks = projector.transform(primitives)
    assert np.all(tracks[1, :, 0] < tracks[0, :, 0])
    assert np.all(tracks[1, :, 1] > tracks[0, :, 1])
    assert np.all(tracks[1, :, 2] < tracks[0, :, 2])
    assert np.all(tracks[1, :, 3] > tracks[0, :, 3])


def test_flow_word_retains_all_chords_and_tracks() -> None:
    chords = np.arange(4 * 10 * 5).reshape(4, 10, 5) % 3
    words = flow_query_words(chords)
    assert words.shape == (4, 50)
    np.testing.assert_array_equal(words.reshape(4, 10, 5), chords)


def test_undefined_first_flow_dynamics_are_neutral() -> None:
    tracks = np.tile(np.linspace(-2.0, 2.0, 30)[:, None, None], (1, 10, 5))
    tokenizer = FlowChordTokenizer().fit(tracks, np.arange(20))
    chords = tokenizer.transform(tracks)
    assert np.all(chords[:, 0, 3] == 1)
    assert np.all(chords[:, 0, 4] == 1)


def test_position_context_keeps_order_separate_from_bag() -> None:
    neutral = np.ones(5, dtype=np.int8)
    a = neutral.copy()
    a[0] = 0
    b = neutral.copy()
    b[0] = 2
    c = neutral.copy()
    c[1] = 2
    d = neutral.copy()
    d[1] = 0
    words = []
    for _ in range(40):
        first = np.tile(neutral, (10, 1))
        first[:3] = np.stack([a, b, c])
        second = np.tile(neutral, (10, 1))
        second[:3] = np.stack([b, a, d])
        words.extend([first, second])
    words = np.asarray(words)
    rows = np.arange(len(words))
    ordered = PositionFactorialContextGrammar(
        max_order=2, min_support=10, ordered=True
    ).fit(words, rows)
    bag = PositionFactorialContextGrammar(
        max_order=2, min_support=10, ordered=False
    ).fit(words, rows)
    code_a, code_b = map(int, chord_codes(np.stack([a, b])))
    ordered_ab, depth = ordered.distribution(2, [code_a, code_b])
    ordered_ba, _ = ordered.distribution(2, [code_b, code_a])
    bag_ab, _ = bag.distribution(2, [code_a, code_b])
    bag_ba, _ = bag.distribution(2, [code_b, code_a])
    assert depth == 2
    assert ordered_ab[1, 2] > ordered_ba[1, 2]
    np.testing.assert_allclose(bag_ab, bag_ba)
    curve, curve_depth = ordered.score_order_curve(words)
    for order in range(3):
        score, depth_score = ordered.score_words(words, max_order=order)
        np.testing.assert_allclose(score, curve[:, order])
        np.testing.assert_allclose(depth_score, curve_depth[:, order])


def test_flow_fl_fl_sy_sy_is_detected_at_any_and_terminal_positions() -> None:
    tracks = np.tile(np.linspace(-2.0, 2.0, 30)[:, None, None], (1, 10, 5))
    tokenizer = FlowChordTokenizer().fit(tracks, np.arange(20))
    assert tokenizer.transform(tracks).shape == tracks.shape
    chords = np.ones((2, 10, 5), dtype=np.int8)
    chords[0, 6:8, 0] = 0
    chords[0, 8:10, 1] = 2
    chords[1, 1:3, 0] = 0
    chords[1, 3:5, 1] = 2
    motifs = flow_motif_indicators(chords)
    np.testing.assert_array_equal(motifs["flow_fl_fl_sy_sy_any"], [1, 1])
    np.testing.assert_array_equal(motifs["flow_fl_fl_sy_sy_terminal"], [1, 0])
