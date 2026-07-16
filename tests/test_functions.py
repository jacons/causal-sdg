"""Tests for causal_sdg.functions.make_dist_from_dataframe2.

Runnable two ways:
    pytest tests/test_functions.py
    python tests/test_functions.py     # no framework needed, prints OK per test
"""
import numpy as np
import pandas as pd

from causal_sdg import make_dist_from_dataframe2 as make_dist


def test_single_parent_empirical_distribution():
    # 'a': child lists explode to [cat, dog, cat] -> cat 2/3, dog 1/3
    # 'b': [horse] -> horse 1.0
    df = pd.DataFrame({"parent": ["a", "a", "b"],
                       "child": [["cat", "dog"], ["cat"], ["horse"]]})
    cpt = make_dist(df, X="parent", Y="child")

    assert cpt.is_valid
    assert cpt.get_dim_input() == 1
    assert set(cpt.y_categories) == {"cat", "dog", "horse"}
    # assert via get_probability so column ordering doesn't matter
    assert cpt.get_probability(("a",), "cat") == 2 / 3
    assert cpt.get_probability(("a",), "dog") == 1 / 3
    assert cpt.get_probability(("a",), "horse") == 0.0
    assert cpt.get_probability(("b",), "horse") == 1.0
    # every row is a valid probability distribution
    for key in cpt.cpt:
        np.testing.assert_allclose(cpt.cpt[key].sum(), 1.0)


def test_str_and_singleton_list_X_are_equivalent():
    df = pd.DataFrame({"parent": ["a", "b"], "child": [["cat"], ["dog"]]})
    a = make_dist(df, X="parent", Y="child")
    b = make_dist(df, X=["parent"], Y="child")
    assert a.cpt.keys() == b.cpt.keys()
    for k in a.cpt:
        np.testing.assert_allclose(a.cpt[k], b.cpt[k])


def test_multi_parent_keys_preserve_X_order():
    df = pd.DataFrame({"p1": ["a", "a", "b"],
                       "p2": ["x", "y", "x"],
                       "child": [["cat"], ["dog"], ["cat", "dog"]]})
    cpt = make_dist(df, X=["p1", "p2"], Y="child")

    assert cpt.get_dim_input() == 2
    # keys are (p1, p2) in the order X was given
    assert ("a", "x") in cpt.cpt
    assert ("b", "x") in cpt.cpt
    np.testing.assert_allclose(cpt.get_distribution(("a", "x")),
                               cpt.cpt[("a", "x")])
    assert cpt.get_probability(("a", "x"), "cat") == 1.0
    assert cpt.get_probability(("b", "x"), "cat") == 0.5
    assert cpt.get_probability(("b", "x"), "dog") == 0.5


def test_marginal_on_builds_fallback():
    df = pd.DataFrame({"p1": ["a", "a", "b"],
                       "p2": ["x", "y", "x"],
                       "child": [["cat"], ["dog"], ["cat", "dog"]]})
    cpt = make_dist(df, X=["p1", "p2"], Y="child", marginal_on="p1")

    assert cpt.idx_feature == 0
    # unseen full config falls back to the marginal over p1='a':
    # (a,x)=[1,0] + (a,y)=[0,1] -> normalized [0.5, 0.5]
    np.testing.assert_allclose(cpt.get_distribution(("a", "unseen")), [0.5, 0.5])
    # exact hit is unchanged
    np.testing.assert_allclose(cpt.get_distribution(("a", "x")),
                               cpt.cpt[("a", "x")])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} passed")
