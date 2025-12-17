from training.features import ensure_feature_order, vectorize_features


def test_vectorize_features_sorted_by_default():
    vec = vectorize_features({"b": 2, "a": 1})
    assert vec == [1.0, 2.0]


def test_ensure_feature_order_prefers_given():
    order = ensure_feature_order(["z", "a"], ["a", "b"])
    assert order == ["z", "a"]
