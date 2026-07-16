"""Tests for causal_sdg.cpt (PTData and CPTData).

Runnable two ways:
    pytest tests/test_cpt.py
    python tests/test_cpt.py     # no framework needed, prints OK per test
"""
import numpy as np

from causal_sdg import CPTData, PTData


# --------------------------------------------------------------------------- #
# PTData
# --------------------------------------------------------------------------- #
def test_ptdata_valid_and_lookups():
    pt = PTData(cpt={(): [0.4, 0.6]}, y_categories=["a", "b"])
    assert pt.is_valid
    assert pt.get_dim_input() == 0          # empty-tuple key -> 0 parents
    assert pt.get_dim_output() == 2
    np.testing.assert_allclose(pt.get_distribution(()), [0.4, 0.6])
    assert pt.get_probability((), "a") == 0.4
    assert pt.get_probability((), "b") == 0.6


def test_ptdata_list_is_coerced_to_array():
    pt = PTData(cpt={(): [0.5, 0.5]}, y_categories=["a", "b"])
    assert isinstance(pt.cpt[()], np.ndarray)


def test_ptdata_invalid_shapes():
    # more than one key
    assert not PTData(cpt={(): [1.0], ("x",): [1.0]}, y_categories=["a"]).is_valid
    # key is not the empty tuple
    assert not PTData(cpt={("x",): [0.5, 0.5]}, y_categories=["a", "b"]).is_valid
    # length mismatch with y_categories
    assert not PTData(cpt={(): [1.0]}, y_categories=["a", "b"]).is_valid
    # does not sum to 1
    assert not PTData(cpt={(): [0.4, 0.4]}, y_categories=["a", "b"]).is_valid


def test_ptdata_get_probability_raises():
    pt = PTData(cpt={(): [0.4, 0.6]}, y_categories=["a", "b"])
    for bad in (("nope",), ()):
        try:
            pt.get_probability(("nope",), "a")  # unknown X
        except KeyError:
            break
    else:
        raise AssertionError("expected KeyError for unknown X")
    try:
        pt.get_probability((), "z")             # unknown Y
        raise AssertionError("expected KeyError for unknown Y")
    except KeyError:
        pass


def test_ptdata_sample_respects_distribution():
    # one-hot distribution -> sampling is deterministic
    pt = PTData(cpt={(): [0.0, 1.0]}, y_categories=["a", "b"])
    assert pt.sample((), elements=1) == ["b"]
    # asking for more elements than non-zero categories is clamped
    assert pt.sample((), elements=5) == ["b"]
    # zero vector -> empty
    pt0 = PTData(cpt={(): [0.5, 0.5]}, y_categories=["a", "b"])
    pt0.cpt[()] = np.zeros(2)
    assert pt0.sample((), elements=1) == []
    for bad in (0, -1):
        try:
            pt.sample((), elements=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_ptdata_copy_is_independent():
    pt = PTData(cpt={(): [0.4, 0.6]}, y_categories=["a", "b"])
    clone = pt.copy()
    clone.cpt[()][0] = 999
    clone.y_categories.append("c")
    assert pt.cpt[()][0] == 0.4
    assert pt.y_categories == ["a", "b"]


# --------------------------------------------------------------------------- #
# CPTData
# --------------------------------------------------------------------------- #
def _basic_cpt():
    return CPTData(
        cpt={("a",): [0.4, 0.6], ("b",): [0.0, 1.0]},
        y_categories=["horse", "cat"],
    )


def test_cptdata_valid_and_distribution():
    c = _basic_cpt()
    assert c.is_valid
    assert c.get_dim_input() == 1
    np.testing.assert_allclose(c.get_distribution(("a",)), [0.4, 0.6])
    np.testing.assert_allclose(c.get_distribution(("b",)), [0.0, 1.0])
    assert c.get_probability(("a",), "cat") == 0.6


def test_cptdata_zero_vector_is_valid():
    # an all-zero row is allowed (means "no data")
    c = CPTData(cpt={("a",): [0.0, 0.0]}, y_categories=["x", "y"])
    assert c.is_valid


def test_cptdata_invalid_rows():
    # non-string key element
    assert not CPTData(cpt={(1,): [0.5, 0.5]}, y_categories=["x", "y"]).is_valid
    # wrong length
    assert not CPTData(cpt={("a",): [1.0]}, y_categories=["x", "y"]).is_valid
    # sums to neither 1 nor 0
    assert not CPTData(cpt={("a",): [0.4, 0.4]}, y_categories=["x", "y"]).is_valid


def test_cptdata_fill_missing():
    c = _basic_cpt()
    # missing key -> zero vector by default
    np.testing.assert_allclose(c.get_distribution(("z",)), [0.0, 0.0])
    # missing key -> uniform when requested
    np.testing.assert_allclose(
        c.get_distribution(("z",), fill_missing="uniform"), [0.5, 0.5]
    )
    try:
        c.get_distribution(("z",), fill_missing="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_cptdata_marginalization_fallback():
    c = CPTData(
        cpt={
            ("a", "x"): [1.0, 0.0],
            ("a", "y"): [0.0, 1.0],
            ("b", "x"): [0.5, 0.5],
        },
        y_categories=["p", "q"],
        marginal_on=0,
    )
    # marginal over parent 0 == 'a' -> ([1,0]+[0,1]) normalized
    np.testing.assert_allclose(c.marginalized_cpt.get_distribution(("a",)), [0.5, 0.5])
    # a known-but-incomplete config falls back to the marginal on parent 0
    np.testing.assert_allclose(c.get_distribution(("a", "unseen")), [0.5, 0.5])
    # an exact hit still returns the exact row
    np.testing.assert_allclose(c.get_distribution(("a", "x")), [1.0, 0.0])
    # parent value absent from the marginal too -> zero fallback
    np.testing.assert_allclose(c.get_distribution(("zzz", "unseen")), [0.0, 0.0])


def test_cptdata_sample_and_copy():
    c = _basic_cpt()
    # ('b',) is one-hot on 'cat' -> deterministic
    assert c.sample(("b",), elements=3) == ["cat"]
    # copy carries the marginalization index and is independent
    m = CPTData(
        cpt={("a", "x"): [1.0, 0.0], ("b", "x"): [0.0, 1.0]},
        y_categories=["p", "q"],
        marginal_on=0,
    )
    clone = m.copy()
    assert clone.idx_feature == 0
    clone.cpt[("a", "x")][0] = 999
    assert m.cpt[("a", "x")][0] == 1.0


def test_sample_frequencies_match_distribution():
    # Draw one element at a time (replace=False only matters for multi-draws) and
    # check the empirical frequencies converge to the declared probabilities.
    np.random.seed(0)
    probs = [0.2, 0.3, 0.5]
    cats = ["x", "y", "z"]

    for data in (
        PTData(cpt={(): probs}, y_categories=cats),
        CPTData(cpt={("k",): probs}, y_categories=cats),
    ):
        key = () if isinstance(data, PTData) and not isinstance(data, CPTData) else ("k",)
        n = 30000
        counts = {c: 0 for c in cats}
        for _ in range(n):
            counts[data.sample(key, elements=1)[0]] += 1
        for c, p in zip(cats, probs):
            assert abs(counts[c] / n - p) < 0.02, (c, counts[c] / n, p)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} passed")
