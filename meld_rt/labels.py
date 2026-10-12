"""label conventions used everywhere. index order matters: it is the order of
every posterior vector in the state and in the feature caches."""

# MELD README index order (and the order of its class weights)
EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMO_INDEX = {e: i for i, e in enumerate(EMOTIONS)}

SENTIMENTS = ["neutral", "positive", "negative"]
SENT_INDEX = {s: i for i, s in enumerate(SENTIMENTS)}

# sensevoice emotion tokens, in the order of its vocabulary rows
SV_EMO_TOKENS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED", "OTHER", "EMO_UNKNOWN"]
SV_EVENT_TOKENS = ["Speech", "BGM", "Laughter", "Applause", "Cry", "Sneeze", "Breath", "Cough", "Sing", "Speech_Noise", "Event_UNK"]

# sensevoice emotion -> MELD emotion (OTHER and EMO_UNKNOWN have no counterpart)
SV_TO_MELD = {"HAPPY": "joy", "SAD": "sadness", "ANGRY": "anger", "NEUTRAL": "neutral",
              "FEARFUL": "fear", "DISGUSTED": "disgust", "SURPRISED": "surprise"}

# valence used by the discrepancy direction rule
VALENCE = {"joy": 1, "neutral": 0, "surprise": 0, "sadness": -1, "anger": -1, "disgust": -1, "fear": -1}

# MELD README class weights for the emotion baseline, index order above
MELD_README_WEIGHTS = [4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0]

# state/json key order for the audio events (lower-case)
EVENT_KEYS = ["speech", "bgm", "laughter", "applause", "cry", "sneeze", "breath", "cough", "sing", "speech_noise", "event_unk"]


def sv_probs_to_meld7(p9) -> list[float]:
    """renormalize a 9-way sensevoice emotion posterior over the seven named classes, MELD order."""
    out = []
    for e in EMOTIONS:
        tok = next(t for t, m in SV_TO_MELD.items() if m == e)
        out.append(float(p9[SV_EMO_TOKENS.index(tok)]))
    s = sum(out)
    return [x / s for x in out] if s > 0 else [1.0 / len(out)] * len(out)
