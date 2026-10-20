import pytest

from finsight.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_basic():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0
    assert recall_at_k(ranked, {"b"}, 2) == 1
    assert mrr_at_k(ranked, {"c"}) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, {"a"}) == 1


def test_evaluate():
    out = evaluate([(["a", "b"], {"a"}), (["a", "b"], {"b"})], ks=(1,))
    assert out["R@1"] == 0.5
    assert out["MRR@10"] == pytest.approx(0.75)
