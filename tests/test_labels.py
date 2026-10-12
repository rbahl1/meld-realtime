from meld_rt import labels


def test_orders_and_maps():
    assert labels.EMOTIONS == ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
    assert len(labels.SV_EMO_TOKENS) == 9 and len(labels.SV_EVENT_TOKENS) == 11 == len(labels.EVENT_KEYS)
    assert set(labels.SV_TO_MELD.values()) == set(labels.EMOTIONS)
    assert len(labels.MELD_README_WEIGHTS) == 7


def test_renormalize():
    p9 = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
    p7 = labels.sv_probs_to_meld7(p9)
    assert abs(sum(p7) - 1) < 1e-9
    assert p7[labels.EMO_INDEX["anger"]] == max(p7)
