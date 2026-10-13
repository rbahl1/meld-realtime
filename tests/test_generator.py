"""streaming reply generator (spec 4) without the LLM files: the prompt and
grounding from state dicts, check_reply, and the decode loop on a tiny onnx
model with the LLM's input and output names and a byte-level bpe tokenizer of a
few words. the tokenizer-level mask tests use the downloaded Qwen tokenizer and
skip when it is absent."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from meld_rt.labels import EMOTIONS
from meld_rt.response import generator as G
from meld_rt.response.lexicon import LEXICON
from meld_rt.state import State, Vision, audio_emotion_dict, emotion7_dict

ROOT = Path(__file__).resolve().parents[1]
QWEN_TOK = ROOT / "models/llm/Qwen2.5-0.5B-Instruct/tokenizer.json"
LOW, HIGH = 0.47, 0.73


def p7(**kw) -> list[float]:
    """7-way vector in MELD order; the rest of the mass spread evenly."""
    rest = (1.0 - sum(kw.values())) / (7 - len(kw))
    return [kw.get(e, rest) for e in EMOTIONS]


def sv9(unknown: float = 0.0, **kw) -> list[float]:
    """9-way sensevoice vector from MELD names (SV_EMO_TOKENS order) via the state helper."""
    d = {e: kw.get(e, 0.0) for e in EMOTIONS} | {"other": 0.0, "unknown": unknown}
    tot = sum(d.values())
    from meld_rt.state import audio_emotion_vec9

    return [v / tot for v in audio_emotion_vec9(d)]


def vstate(said="I can not believe you did that to my car", text=None, audio=None, face=None, label="anger", conf=0.62,
           calibrated=True, degraded=(), flag=False, direction="none", face_flag=False, face_direction="none",
           vision_mode=True) -> dict:
    s = State.empty("t", 3, vision_mode=vision_mode)
    s.asr.text, s.asr.conf = said, 0.9
    s.text.emotion = emotion7_dict(text or p7(anger=0.6))
    s.audio.emotion = audio_emotion_dict(audio or sv9(unknown=0.7, anger=0.3))
    s.fused.label, s.fused.confidence, s.fused.calibrated = label, conf, calibrated
    s.degraded = list(degraded)
    s.discrepancy.flag, s.discrepancy.direction = flag, direction
    if vision_mode:
        s.vision = Vision(emotion=emotion7_dict(face), present=True, rule="asd") if face is not None else Vision(rule="asd")
        s.discrepancy.face_flag, s.discrepancy.face_direction = face_flag, face_direction
    return s.to_dict()


def user(msgs) -> str:
    return msgs[-1]["content"]


# ---------------------------------------------------------------- prompt

def test_full_state_prompt_grounding_and_allowed():
    st = vstate(text=p7(anger=0.6), audio=sv9(unknown=0.5, sadness=0.45, anger=0.05), face=p7(joy=0.41), conf=0.62)
    msgs, grounding, allowed = G.build_messages(st, LOW, HIGH)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == G.SYSTEM_PROMPT
    lines = user(msgs).split("\n")
    assert lines[0] == 'They said: "I can not believe you did that to my car"'
    assert lines[1:4] == ["The words suggest anger.", "The voice suggests sadness.", "The face suggests joy."]
    assert "Overall reading: anger, medium confidence." in lines
    assert lines[-1] == "Reply to them in one or two short sentences. You may name only these feelings: sadness, joy, anger."
    # the fused label plus every modality's top class at 0.4 or more, in EMOTIONS order
    assert allowed == ["sadness", "joy", "anger"]
    assert grounding[0] == "fused.label=anger@0.62"
    assert "asr.text='I can not believe you did that to my car'" in grounding
    assert {"text.emotion=anger@0.60", "audio.emotion=sadness@0.45", "vision.emotion=joy@0.41"} <= set(grounding)


def test_voice_uses_the_raw_mass_so_unknown_counts_against_it():
    # renormalized over the named classes anger would be 0.75; its raw mass is 0.15
    st = vstate(audio=sv9(unknown=0.8, anger=0.15, neutral=0.05), face=p7(neutral=0.5))
    msgs, grounding, allowed = G.build_messages(st, LOW, HIGH)
    assert "The voice gives no clear emotion." in user(msgs)
    assert "The face looks neutral." in user(msgs)
    assert "audio.emotion=anger@0.15" in grounding
    assert allowed == ["neutral", "anger"]


def test_confidence_words_and_uncalibrated_cap():
    def word(conf, cal=True):
        return next(ln for ln in user(G.build_messages(vstate(conf=conf, calibrated=cal), LOW, HIGH)[0]).split("\n")
                    if ln.startswith("Overall"))

    assert word(0.47) == "Overall reading: anger, medium confidence."
    assert word(0.73) == "Overall reading: anger, high confidence."
    assert word(0.9, cal=False) == "Overall reading: anger, medium confidence."
    assert word(0.46) == "Overall reading: unclear, low confidence."


def test_low_confidence_asks_and_allows_nothing():
    msgs, grounding, allowed = G.build_messages(vstate(conf=0.3, face=p7(anger=0.9)), LOW, HIGH)
    assert allowed == []
    assert grounding[0] == "fused.confidence=0.30<0.47"
    assert not any(g.startswith("fused.label") for g in grounding)
    assert user(msgs).endswith("Reply with one short question about what they said, and do not name any feeling.")
    assert "anger," not in user(msgs).split("Overall")[1]


def test_asr_failure_and_empty_transcript():
    for st, g0 in ((vstate(said="", degraded=["asr_failed"], conf=0.0), "degraded=asr_failed"),
                   (vstate(said="uh", degraded=["low_asr_conf"], conf=0.0), "degraded=low_asr_conf"),
                   (vstate(said="", conf=0.0), "asr.text=''")):
        msgs, grounding, allowed = G.build_messages(st, LOW, HIGH)
        u = user(msgs)
        assert u.startswith("The words were not caught.") and "They said" not in u and "The words suggest" not in u
        assert u.endswith("Ask them to say it again, and do not name any feeling.")
        assert allowed == [] and g0 in grounding


def test_no_face_and_core_state():
    msgs, grounding, _ = G.build_messages(vstate(face=None), LOW, HIGH)
    assert "No face was seen." in user(msgs) and "vision.present=false" in grounding
    core = vstate(vision_mode=False)
    assert "vision" not in core
    msgs, grounding, _ = G.build_messages(core, LOW, HIGH)
    assert "No face was seen." in user(msgs) and not any(g.startswith("vision") for g in grounding)


CLEAR_VOICE = sv9(unknown=0.3, sadness=0.6, anger=0.1)        # raw sadness mass 0.6: "The voice suggests sadness."
UNCLEAR_VOICE = sv9(unknown=0.8, anger=0.15, neutral=0.05)     # raw mass below 0.4; renormalized anger 0.75


def test_disagreement_sentences_only_when_flagged():
    st = vstate(audio=CLEAR_VOICE, face=p7(sadness=0.6), flag=True, direction="tone_more_negative", face_flag=True,
                face_direction="face_more_negative")
    msgs, grounding, _ = G.build_messages(st, LOW, HIGH)
    u = user(msgs)
    assert "The voice suggests sadness." in u
    assert "The voice sounds more negative than the words." in u and "The face looks more negative than the words." in u
    assert "discrepancy.direction=tone_more_negative" in grounding and "discrepancy.face_direction=face_more_negative" in grounding
    st = vstate(face=p7(sadness=0.6), direction="tone_more_negative", face_direction="face_more_negative")
    u = user(G.build_messages(st, LOW, HIGH)[0])
    assert "more negative" not in u
    for d, line in [*G.DIRECTION_VOICE.items()]:
        assert line in user(G.build_messages(vstate(audio=CLEAR_VOICE, flag=True, direction=d), LOW, HIGH)[0])
    for d, line in [*G.DIRECTION_FACE.items()]:
        assert line in user(G.build_messages(vstate(face=p7(joy=0.5), face_flag=True, face_direction=d), LOW, HIGH)[0])


def test_unclear_voice_and_a_words_voice_direction_do_not_contradict():
    # the voice line reads the raw mass (no clear emotion here), the direction the renormalized posterior
    for d in G.DIRECTION_VOICE:
        st = vstate(text=p7(joy=0.7), audio=UNCLEAR_VOICE, label="joy", flag=True, direction=d)
        msgs, grounding, allowed = G.build_messages(st, LOW, HIGH)            # GAP_STYLE lean
        u = user(msgs).split("\n")
        assert G.VOICE_LEAN[d] in u and "The voice gives no clear emotion." not in u
        assert not any(line in u for line in G.DIRECTION_VOICE.values())
        assert f"discrepancy.direction={d}" in grounding and "audio.emotion=anger@0.15" in grounding
        assert allowed == ["joy"]
        msgs, grounding, allowed = G.build_messages(st, LOW, HIGH, gap_style="direction_posterior")
        u = user(msgs).split("\n")
        assert u[2] == "The voice suggests anger." and G.DIRECTION_VOICE[d] in u
        assert "audio.emotion7=anger@0.75" in grounding and allowed == ["joy", "anger"]
    # unflagged: both styles read the raw mass
    st = vstate(text=p7(joy=0.7), audio=UNCLEAR_VOICE, label="joy")
    for gap in G.GAP_STYLES:
        assert "The voice gives no clear emotion." in user(G.build_messages(st, LOW, HIGH, gap_style=gap)[0])
    with pytest.raises(ValueError):
        G.build_messages(st, LOW, HIGH, gap_style="loud")


def test_unclear_face_and_a_words_face_direction_do_not_contradict():
    # the face line reads the calibrated face posterior, the direction the raw fer posterior (vision.raw)
    raw = p7(sadness=0.6)
    for d in G.DIRECTION_FACE:
        st = vstate(text=p7(joy=0.7), audio=CLEAR_VOICE, face=p7(sadness=0.3), label="joy", face_flag=True, face_direction=d)
        st["vision"]["raw"] = dict(zip(EMOTIONS, raw))
        msgs, grounding, allowed = G.build_messages(st, LOW, HIGH)
        u = user(msgs).split("\n")
        assert u[3] == G.FACE_LEAN[d] and "The face gives no clear emotion." not in u
        assert not any(line in u for line in G.DIRECTION_FACE.values())
        assert f"discrepancy.face_direction={d}" in grounding and "vision.emotion=sadness@0.30" in grounding
        assert allowed == ["sadness", "joy"]                                   # sadness from the voice
        msgs, grounding, allowed = G.build_messages(st, LOW, HIGH, gap_style="direction_posterior")
        u = user(msgs).split("\n")
        assert u[3] == "The face suggests sadness." and G.DIRECTION_FACE[d] in u and "vision.raw=sadness@0.60" in grounding
    # a clear face keeps the two sentences in both styles
    st = vstate(face=p7(sadness=0.6), face_flag=True, face_direction="face_more_negative")
    for gap in G.GAP_STYLES:
        u = user(G.build_messages(st, LOW, HIGH, gap_style=gap)[0]).split("\n")
        assert "The face suggests sadness." in u and "The face looks more negative than the words." in u


def test_neutral_only_names_no_feeling():
    st = vstate(text=p7(neutral=0.7), audio=sv9(unknown=0.9, neutral=0.1), face=None, label="neutral", conf=0.6)
    msgs, _, allowed = G.build_messages(st, LOW, HIGH)
    assert allowed == ["neutral"]
    assert user(msgs).endswith("Reply to them in one or two short sentences, and do not name any feeling.")


def test_transcript_cleanup_and_cut():
    u = user(G.build_messages(vstate(said='He said “no” — it’s "over"'), LOW, HIGH)[0])
    assert u.split("\n")[0] == "They said: \"He said 'no' - it's 'over'\""
    long = " ".join(f"w{i}" for i in range(100))
    first = user(G.build_messages(vstate(said=long), LOW, HIGH)[0]).split("\n")[0]
    assert first.endswith('w59 ..."') and "w60" not in first


def test_warmup_state_and_default_high_conf(tmp_path):
    for vm in (False, True):
        msgs, _, allowed = G.build_messages(State.empty("warmup", 0, vision_mode=vm).to_dict(), 0.45)
        assert allowed == [] and msgs[-1]["role"] == "user"
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"thresholds": {"high_conf": 0.73}, "vision": {"thresholds": {"high_conf": 0.66}}}))
    assert G.default_high_conf(False, m) == 0.73 and G.default_high_conf(True, m) == 0.66
    m2 = tmp_path / "m2.json"
    m2.write_text(json.dumps({"thresholds": {"high_conf": 0.73}}))
    assert G.default_high_conf(True, m2) == 0.73
    assert G.default_high_conf(True, tmp_path / "missing.json") == 0.70


def test_examples_follow_their_own_rules(bench):
    assert 2 <= len(G.EXAMPLES) <= 3
    for ev, rep, allowed in G.EXAMPLES:
        assert ev.startswith("They said:") and "Overall reading:" in ev
        assert G.check_reply(rep, allowed)["ok"], rep
        # "do not repeat their words back to them": no run the echo guard rolls back, and no parrot
        grams = set(G.transcript_ngrams([{"role": "user", "content": ev}], G.ECHO_NGRAM))
        w = [x.lower() for x in G._WORDS.findall(rep)]
        assert grams and not grams & {tuple(w[i:i + G.ECHO_NGRAM]) for i in range(len(w) - G.ECHO_NGRAM + 1)}, rep
        said = re.match(r'They said: "(.*)"', ev).group(1)
        assert bench.longest_shared_run(rep, said) < G.ECHO_NGRAM and bench.parrot(rep, said)["transcript"] is False, rep
    assert G.EXAMPLES[-1][2] == []          # one low-confidence example that asks
    for ev, rep, _ in G.EXAMPLES:
        assert ev in G.SYSTEM_PROMPT and rep in G.SYSTEM_PROMPT


def test_example_turns_and_chatml_split():
    st = vstate(face=p7(anger=0.5))
    msgs, _, _ = G.build_messages(st, LOW, HIGH, examples_in="turns")
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * len(G.EXAMPLES) + ["user"]
    assert msgs[0]["content"] == G.RULES
    prefix, rest = G.chatml(msgs)
    assert prefix.startswith("<|im_start|>system\n") and prefix.endswith("<|im_end|>\n")
    assert rest == f"<|im_start|>user\n{msgs[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"
    # the prefix does not depend on the state, so its kv cache is shared by every reply
    other = G.build_messages(vstate(said="hello there", conf=0.2), LOW, HIGH, examples_in="turns")[0]
    assert G.chatml(other)[0] == prefix
    with pytest.raises(ValueError):
        G.build_messages(st, LOW, HIGH, examples_in="nowhere")
    with pytest.raises(ValueError):
        G.chatml([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------- check_reply

def test_check_reply():
    c = G.check_reply("You sound angry about the car. What happened?", ["anger"])
    assert c == {"n_words": 8, "over_length": False, "exclamation": False, "forbidden_words": [], "empty": False,
                 "meta_words": [], "prompt_echo": [], "non_ascii": False, "non_printable": False, "ok": True}
    c = G.check_reply("You sound Angry and sad!", ["sadness"])
    assert c["forbidden_words"] == ["Angry"] and c["exclamation"] and not c["ok"]
    assert G.check_reply("", [])["empty"] and not G.check_reply("  ", [])["ok"]
    assert G.check_reply(" ".join(["word"] * 21), [])["over_length"]
    assert not G.check_reply(" ".join(["word"] * 20), [])["over_length"]
    assert G.check_reply("The model says the audio is calm.", [])["meta_words"] == ["model", "audio"]
    assert G.check_reply("Overall reading: joy, high confidence.", ["joy"])["prompt_echo"] == ["overall reading", "high confidence"]
    assert G.check_reply("You are a small desk robot.", [])["prompt_echo"] == ["desk robot"]      # the system prompt copied
    assert G.check_reply("You sound tired — rest.", [])["non_ascii"]
    assert G.check_reply("You sound happy.", [])["forbidden_words"] == ["happy"]
    for t in ("You sound\x0btired.", "Okay\x00", "One\x0cTwo", "A\x7f"):
        c = G.check_reply(t, [])
        assert c["non_printable"] and not c["non_ascii"] and not c["ok"], repr(t)
    assert not G.check_reply("You sound tired. Rest, then talk.", [])["non_printable"]


# ---------------------------------------------------------------- decode loop on a tiny model

WORDS = ["You", "Ġsound", "Ġangry", "Ġfr", "ustr", "ated", "Ġtired", "Ġok", "!", ".", "They", "Ġsaid", "The", "Ġvoice", "Ġis"]


def _bpe_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(G._byte_decoder(), key=lambda ch: G._byte_decoder()[ch])
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    merges = []
    for w in WORDS:
        for k in range(2, len(w) + 1):
            left, right = w[: k - 1], w[k - 1]
            if (left, right) not in merges:
                merges.append((left, right))
            vocab.setdefault(w[:k], len(vocab))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    return tok


def _fake_lm(tmp_path: Path, rows: dict[str, list[str]], extra_logits: int = 3) -> Path:
    """model folder with tokenizer.json and onnx/model_int8.onnx. logits depend on the last
    input token only: rows[t] lists the next tokens best first. the kv cache is a
    per-token embedding concatenated to the past, as the real graph's present outputs."""
    import onnx
    from onnx import TensorProto, helper

    tok = _bpe_tokenizer()
    vocab = tok.get_vocab(with_added_tokens=True)
    v = len(vocab) + extra_logits                 # padded output rows, as the real lm head
    table = np.zeros((v, v), dtype=np.float32)
    for t, nxt in rows.items():
        for rank, n in enumerate(nxt):
            table[vocab[t], vocab[n]] = 10.0 - rank
    kvt = np.stack([np.arange(v, dtype=np.float32), np.ones(v, np.float32)], axis=1)
    eye = np.eye(v, dtype=np.float32)
    inits = [helper.make_tensor("table", TensorProto.FLOAT, table.shape, table.ravel()),
             helper.make_tensor("onehot", TensorProto.FLOAT, eye.shape, eye.ravel()),
             helper.make_tensor("kvt", TensorProto.FLOAT, kvt.shape, kvt.ravel()),
             helper.make_tensor("ax1", TensorProto.INT64, [1], [1])]
    # one-hot hidden state times the table: an lm head MatMul over an activation, as in the real graph
    nodes = [helper.make_node("Gather", ["onehot", "input_ids"], ["hidden"]),
             helper.make_node("MatMul", ["hidden", "table"], ["logits"]),
             helper.make_node("Gather", ["kvt", "input_ids"], ["e"]),
             helper.make_node("Unsqueeze", ["e", "ax1"], ["e4"])]
    ins = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "s"]),
           helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "t"]),
           helper.make_tensor_value_info("position_ids", TensorProto.INT64, [1, "s"])]
    outs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "s", v])]
    for i in range(2):
        for kind in ("key", "value"):
            p, q = f"past_key_values.{i}.{kind}", f"present.{i}.{kind}"
            ins.append(helper.make_tensor_value_info(p, TensorProto.FLOAT, [1, 1, "p", 2]))
            outs.append(helper.make_tensor_value_info(q, TensorProto.FLOAT, [1, 1, "q", 2]))
            nodes.append(helper.make_node("Concat", [p, "e4"], [q], axis=2))
    # the mask and positions feed an op so they stay graph inputs
    nodes += [helper.make_node("Shape", ["attention_mask"], ["ms"]), helper.make_node("Shape", ["position_ids"], ["ps"])]
    outs += [helper.make_tensor_value_info("ms", TensorProto.INT64, [2]), helper.make_tensor_value_info("ps", TensorProto.INT64, [2])]
    graph = helper.make_graph(nodes, "fake_lm", ins, outs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    d = tmp_path / "fake-llm"
    (d / "onnx").mkdir(parents=True)
    onnx.save(model, str(d / "onnx" / "model_int8.onnx"))
    tok.save(str(d / "tokenizer.json"))
    return d


# after the assistant header ("\n" is the last prompt token) the model says
# "You sound angry." unless the mask steers it to " fr ustr ated" (frustrated, three
# pieces, not a whole-word token) and the word guard then to " tired"
ROWS = {"Ċ": ["You"], "You": ["Ġsound"], "Ġsound": ["Ġangry", "Ġfr", "Ġtired"], "Ġangry": ["."],
        "Ġfr": ["ustr"], "ustr": ["ated"], "ated": ["."], "Ġtired": ["!", "."], ".": ["<|im_end|>"]}


@pytest.fixture
def gen(tmp_path):
    return G.ReplyGenerator(_fake_lm(tmp_path, ROWS), threads=1)


MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_graph_names_are_read(gen):
    assert gen.past_names == ["past_key_values.0.key", "past_key_values.0.value", "past_key_values.1.key", "past_key_values.1.value"]
    assert gen.present_names == [n.replace("past_key_values", "present") for n in gen.past_names]
    assert gen.kv_heads == 1 and gen.head_dim == 2 and gen.n_layers == 2 and gen.has_position_ids
    assert gen.n_logits == len(gen.tok.get_vocab(with_added_tokens=True)) + 3
    assert gen.name == "fake-llm/model_int8.onnx" and gen.last_position_logits


def test_last_position_logits_match_the_full_graph(tmp_path):
    import onnxruntime as ort

    d = _fake_lm(tmp_path, ROWS)
    path = d / "onnx" / "model_int8.onnx"
    full = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    last = ort.InferenceSession(G.last_position_model(path), providers=["CPUExecutionProvider"])
    ids = np.array([[5, 9, 200, 7]], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones((1, 4), np.int64), "position_ids": np.arange(4)[None]}
    feed |= {n.name: np.zeros((1, 1, 0, 2), np.float32) for n in full.get_inputs() if n.name.startswith("past")}
    a, b = full.run(["logits", "present.0.key"], feed), last.run(["logits", "present.0.key"], feed)
    assert a[0].shape[1] == 4 and b[0].shape[1] == 1
    np.testing.assert_array_equal(a[0][:, -1], b[0][:, 0])
    np.testing.assert_array_equal(a[1], b[1])
    g = G.ReplyGenerator(d, threads=1, last_logits=False)
    assert not g.last_position_logits
    assert g.generate(MSGS, [], None).text == G.ReplyGenerator(d, threads=1).generate(MSGS, [], None).text


def test_masks_of_the_small_tokenizer(gen):
    v = gen.tok.get_vocab(with_added_tokens=True)
    assert v["Ġangry"] in gen.masks.by_class["anger"].tolist()
    assert v["Ġfr"] not in gen.masks.by_class["anger"].tolist()       # a piece, left to the word guard
    reg = set(gen.masks.register.tolist())
    assert v["!"] in reg and v["<|im_start|>"] in reg and v["Ċ"] in reg
    assert v["<|im_end|>"] not in reg and v["<|endoftext|>"] not in reg and v["Ġsound"] not in reg
    assert set(range(len(v), gen.n_logits)) <= reg                      # padding rows
    assert set(gen.masks.stop.tolist()) == {v["<|im_end|>"], v["<|endoftext|>"]}
    ch = {b: c for c, b in G._byte_decoder().items()}                   # byte -> byte-level token
    assert all(v[ch[b]] in reg for b in [*range(0x20), 0x7f, 0x80, 0xff])  # ascii control bytes, del, non-ascii
    assert not any(v[ch[b]] in reg for b in (0x20, ord("a"), ord("."), ord("?"), ord("~")))


def test_allowed_class_streams_words(gen):
    words = []
    r = gen.generate(MSGS, ["anger"], words.append)
    assert r.text == "You sound angry." and words == ["You", " sound", " angry."] == r.deltas
    assert r.stop_reason == "eos" and r.masked_steps == 0 and r.backtracks == 0 and r.n_tokens == 4 and r.steps == 5
    # one forward pass per token after the first token, the pass that yields the end token included
    assert r.decode_passes == 4 and r.rework_passes == 0
    assert r.first_token_ms > 0 and r.total_ms >= r.first_token_ms and r.prompt_tokens > 0


def test_mask_then_word_guard_then_register_mask(gen):
    words = []
    r = gen.generate(MSGS, [], words.append)
    # " angry" is masked; " fr ustr ated" spells frustrated and is rolled back; "!" is masked
    assert r.text == "You sound tired." and words == ["You", " sound", " tired."]
    # the step after " sound" is decided twice (before and after the rollback), and both times the mask acts
    assert r.masked_steps == 2 and r.backtracks == 1 and r.register_masked_steps == 1 and r.steps == 9
    # passes after You, sound, fr, ustr, ated (the last three undone) and after tired and the final "."
    assert r.word_backtracks == 1 and r.decode_passes == 7 and r.rework_passes == 3
    assert G.check_reply(r.text, [])["ok"]
    assert gen.generate(MSGS, [], None).text == r.text                  # deterministic, prefix cache reused
    assert len(gen._prefix) == 1


def test_guard_phrase_is_rolled_back_and_held_back(tmp_path):
    # "They said ..." copies the notes: rolled back to "They", then "The voice is ..." waits a word
    rows = {"Ċ": ["They", "The"], "They": ["Ġsaid"], "Ġsaid": ["."], "The": ["Ġvoice"], "Ġvoice": ["Ġis"], "Ġis": ["Ġok"],
            "Ġok": ["."], ".": ["<|im_end|>"]}
    g = G.ReplyGenerator(_fake_lm(tmp_path, rows), threads=1)
    words = []
    seen_at = []

    def on_delta(w):
        words.append(w)
        seen_at.append(len(words))

    r = g.generate(MSGS, ["anger"], on_delta)
    assert r.text == "The voice is ok." and r.backtracks == 1 and words == ["The", " voice", " is", " ok."]
    # no rollback left: the reply ends at the last word that passed. "They" was held back, so nothing
    # was sent, the reply is empty and the service answers with the bank line (fallback empty_reply)
    g0 = G.ReplyGenerator(g.onnx_path, threads=1, max_backtracks=0)
    words0 = []
    r0 = g0.generate(MSGS, ["anger"], words0.append)
    assert r0.text == "" and words0 == [] and r0.stop_reason == "guard" and r0.backtracks == 0


def test_transcript_ngrams():
    msgs = [{"role": "user", "content": 'They said: "I am not, going to go"\nThe words sound neutral.'}]
    assert G.transcript_ngrams(msgs, 4) == (("i", "am", "not", "going"), ("am", "not", "going", "to"), ("not", "going", "to", "go"))
    assert G.transcript_ngrams(msgs, 0) == ()
    assert G.transcript_ngrams([{"role": "user", "content": "The words were not caught."}], 4) == ()
    st = G.build_messages(vstate(said="one two three four five"), LOW, HIGH)[0]
    assert len(G.transcript_ngrams(st, G.ECHO_NGRAM)) == 2


def test_echo_of_the_transcript_is_rolled_back(tmp_path):
    rows = {"Ċ": ["You", "The"], "You": ["Ġsound"], "Ġsound": ["Ġtired"], "Ġtired": ["Ġok"], "Ġok": ["."], "The": ["Ġvoice"],
            "Ġvoice": ["Ġis"], "Ġis": ["Ġok"], ".": ["<|im_end|>"]}
    d = _fake_lm(tmp_path, rows)
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": 'They said: "you sound tired ok"'}]
    words = []
    r = G.ReplyGenerator(d, threads=1).generate(msgs, [], words.append)
    assert r.text == "The voice is ok." and r.backtracks == 1 and words == ["The", " voice", " is", " ok."]
    assert G.ReplyGenerator(d, threads=1, echo_ngram=0).generate(msgs, [], None).text == "You sound tired ok."
    # three words of the transcript are not an echo
    msgs3 = [msgs[0], {"role": "user", "content": 'They said: "you sound tired"'}]
    assert G.ReplyGenerator(d, threads=1).generate(msgs3, [], None).text == "You sound tired ok."


def test_hold_back_waits_only_inside_a_phrase(tmp_path):
    # "The voice sound" could still become "the voice sounds", so "The" waits until " ok" arrives
    rows = {"Ċ": ["The"], "The": ["Ġvoice"], "Ġvoice": ["Ġsound"], "Ġsound": ["Ġok"], "Ġok": ["."], ".": ["<|im_end|>"]}
    g = G.ReplyGenerator(_fake_lm(tmp_path, rows), threads=1)
    calls = []
    run = g._run

    def counted(*a):
        calls.append(1)
        return run(*a)

    g._run = counted
    got = []
    r = g.generate(MSGS, [], lambda w: got.append((w, len(calls))))
    assert r.text == "The voice sound ok."
    # model calls: 1 prefix, 1 rest, then one per generated token
    assert got == [("The", 5), (" voice", 5), (" sound", 5), (" ok.", 7)]
    calls.clear()
    got.clear()
    g.generate(MSGS, ["anger"], lambda w: got.append((w, len(calls))))     # prefix cached now
    assert got[0] == ("The", 4)


def test_no_rollback_budget_ends_the_reply_at_the_last_sent_word(tmp_path):
    # as at the token budget: the forbidden word is dropped, never streamed, whether the reply
    # would have gone on or ended right after it
    for rows in (ROWS, {**ROWS, "ated": ["<|im_end|>"]}):
        d = tmp_path / str(len(rows["ated"][0]))
        g = G.ReplyGenerator(_fake_lm(d, rows), threads=1, max_backtracks=0)
        words = []
        r = g.generate(MSGS, [], words.append)
        assert r.text == "You sound" and words == ["You", " sound"] and r.stop_reason == "guard" and r.backtracks == 0
        assert G.check_reply(r.text, [])["ok"]


def test_lexicon_words_and_phrases_have_separate_budgets(tmp_path):
    # "They said." spends the phrase budget; "frustrated" still gets a rollback from the word budget
    rows = {"Ċ": ["They", "You"], "They": ["Ġsaid"], "Ġsaid": ["."], "You": ["Ġsound"], "Ġsound": ["Ġfr", "Ġtired"],
            "Ġfr": ["ustr"], "ustr": ["ated"], "ated": ["."], "Ġtired": ["."], ".": ["<|im_end|>"]}
    g = G.ReplyGenerator(_fake_lm(tmp_path, rows), threads=1, max_backtracks=1)
    r = g.generate(MSGS, [], None)
    assert r.text == "You sound tired." and r.backtracks == 2 and r.word_backtracks == 1 and r.stop_reason == "eos"


def test_first_word_with_a_leading_space_can_be_rolled_back(tmp_path):
    # the first token is " fr" (leading space): " frustrated" is still unsent, so it is rolled back
    rows = {"Ċ": ["Ġfr", "You"], "Ġfr": ["ustr"], "ustr": ["ated"], "ated": ["."], "You": ["Ġsound"], "Ġsound": ["Ġtired"],
            "Ġtired": ["."], ".": ["<|im_end|>"]}
    words = []
    r = G.ReplyGenerator(_fake_lm(tmp_path / "a", rows), threads=1).generate(MSGS, [], words.append)
    assert r.text == "You sound tired." and r.backtracks == 1 and words == ["You", " sound", " tired."]
    # the leading space is not sent: the first delta starts at the word
    rows = {"Ċ": ["Ġsound"], "Ġsound": ["Ġtired"], "Ġtired": ["."], ".": ["<|im_end|>"]}
    words = []
    r = G.ReplyGenerator(_fake_lm(tmp_path / "b", rows), threads=1).generate(MSGS, [], words.append)
    assert r.text == "sound tired." and words == ["sound", " tired."]


def test_budget_deadline_and_exceptions(gen):
    r = gen.generate(MSGS, ["anger"], None)
    g2 = G.ReplyGenerator(gen.onnx_path, threads=1, max_new_tokens=2)
    r2 = g2.generate(MSGS, ["anger"], None)
    assert r2.stop_reason == "max_tokens" and r2.text == "You sound" and r.text.startswith(r2.text)
    r3 = gen.generate(MSGS, ["anger"], None, deadline_s=0.0)
    assert r3.stop_reason == "deadline" and r3.text == "" and r3.n_tokens == 0

    def boom(_w):
        raise KeyError("stop")

    with pytest.raises(KeyError):
        gen.generate(MSGS, ["anger"], boom)


def test_cut_word_at_the_token_budget_is_dropped(tmp_path):
    g = G.ReplyGenerator(_fake_lm(tmp_path, ROWS), threads=1, max_new_tokens=4, max_backtracks=0)
    r = g.generate(MSGS, [], None)          # You sound fr ustr | budget: "frustr" would be the rest
    assert r.stop_reason == "max_tokens" and r.text == "You sound frustr"
    g = G.ReplyGenerator(g.onnx_path, threads=1, max_new_tokens=5, max_backtracks=0)
    r = g.generate(MSGS, [], None)          # "frustrated" is finished at the budget and forbidden
    assert r.text == "You sound"


def test_resolve_model(tmp_path):
    d = _fake_lm(tmp_path, ROWS)
    assert G.resolve_model(d) == (d / "onnx" / "model_int8.onnx", d)
    assert G.resolve_model(d / "onnx" / "model_int8.onnx") == (d / "onnx" / "model_int8.onnx", d)
    with pytest.raises(FileNotFoundError):
        G.resolve_model(d, "model_q4.onnx")
    with pytest.raises(FileNotFoundError):
        G.resolve_model(tmp_path / "nothing")


# ---------------------------------------------------------------- the real tokenizer

needs_qwen = pytest.mark.skipif(not QWEN_TOK.exists(), reason="LLM tokenizer not fetched (tools/fetch_models.py)")


@pytest.fixture(scope="module")
def qwen():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(QWEN_TOK))
    return tok, G.token_masks(tok, 151936)


@needs_qwen
def test_qwen_whole_word_masks(qwen):
    tok, m = qwen
    for c, ws in LEXICON.items():
        for i in m.by_class[c].tolist():
            word = tok.decode([i]).strip().lower()
            assert word in ws, (c, word)
    anger = set(m.by_class["anger"].tolist())
    for v in (" angry", " Angry", " furious", " annoyed", " mad", "Sad"):
        ids = tok.encode(v, add_special_tokens=False).ids
        assert len(ids) == 1 and ids[0] in anger | set(m.by_class["sadness"].tolist()), v
    assert m.n_split_variants > 0
    assert len(m.by_class["neutral"]) == 0


@needs_qwen
def test_qwen_register_mask(qwen):
    tok, m = qwen
    reg = set(m.register.tolist())
    v = tok.get_vocab(with_added_tokens=True)
    assert v["<|im_end|>"] not in reg and v["<|endoftext|>"] not in reg and v["<|im_start|>"] in reg
    for s in ("!", " !", "\n", " ’"):
        for i in tok.encode(s, add_special_tokens=False).ids[:1]:
            assert i in reg, s
    # ascii control bytes other than newline and tab, and del: one token each, all masked
    for b in [*range(0x20), 0x7f]:
        ids = tok.encode(chr(b), add_special_tokens=False).ids
        assert len(ids) == 1 and ids[0] in reg, hex(b)
    for s in (" the", " you", ".", ",", "?", " sound", "'s"):
        ids = tok.encode(s, add_special_tokens=False).ids
        assert len(ids) == 1 and ids[0] not in reg, s
    assert set(range(len(v), 151936)) <= reg
    assert set(m.stop.tolist()) == {v["<|im_end|>"], v["<|endoftext|>"]}


@needs_qwen
def test_qwen_prompt_splits_at_a_token_boundary(qwen):
    tok, _ = qwen
    for where in G.EXAMPLE_PLACES:
        msgs, _, _ = G.build_messages(vstate(face=p7(anger=0.5)), LOW, HIGH, examples_in=where)
        prefix, rest = G.chatml(msgs)
        whole = tok.encode(prefix + rest, add_special_tokens=False).ids
        a, b = tok.encode(prefix, add_special_tokens=False).ids, tok.encode(rest, add_special_tokens=False).ids
        assert whole == a + b
        assert tok.decode(b[-3:], skip_special_tokens=False) == "<|im_start|>assistant\n"


# ---------------------------------------------------------------- bench states, summary and choice rule

sys_path_tools = str(ROOT / "tools")


@pytest.fixture(scope="module")
def bench():
    import sys

    sys.path.insert(0, sys_path_tools)
    sys.path.insert(0, str(ROOT / "tests"))
    import bench_generator

    return bench_generator


def test_parrot_and_shared_run(bench):
    assert bench.longest_shared_run("You say it went fine, but", "it went fine, I guess") == 3
    p = bench.parrot("So what do you want to happen next?", "hello")
    assert p["example"] and not p["transcript"] and p["example_run"] >= bench.PARROT_RUN


def _reply(ok=True, n_tokens=10, first=100.0, total=600.0, masked=0, steps=11, allowed=("joy",), names=True, passes=10, rework=0):
    return {"key": "1_1", "variant": "gold", "text": "x", "n_tokens": n_tokens, "first_token_ms": first, "first_word_ms": first + 20,
            "total_ms": total, "stop_reason": "eos", "masked_steps": masked, "steps": steps, "backtracks": 0, "word_backtracks": 0,
            "decode_passes": passes, "rework_passes": rework,
            "register_masked_steps": 0, "prompt_tokens": 50, "allowed": list(allowed), "names_fused": names,
            "parrot": {"transcript": False, "example": False},
            "checks": {"ok": ok, "n_words": 6, "over_length": False, "exclamation": not ok, "empty": False, "non_ascii": False,
                       "forbidden_words": [], "meta_words": [], "prompt_echo": []}}


def test_summarize(bench):
    rs = [_reply(), _reply(ok=False, masked=2, names=False), _reply(allowed=(), names=None)]
    s = bench.summarize(rs)
    assert s["n"] == 3 and s["check_pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["decode_passes_per_s"] == pytest.approx(30 / 1.5)          # 10 passes per 0.5 s: the end token's counts
    assert s["decode_tokens_per_s"] == pytest.approx(27 / 1.5)          # the final-reply rate: 9 tokens per 0.5 s
    assert s["rework_passes"] == {"n": 0, "share": 0.0}
    assert s["masked_step_rate"] == pytest.approx(2 / 33, abs=1e-4) and s["replies_with_masked_step"] == pytest.approx(1 / 3, abs=1e-4)
    # rollback rework does not lower the decode rate: 6 more passes in 0.3 s more, at the same speed
    s2 = bench.summarize([_reply(), _reply(total=900.0, passes=16, rework=6)])
    assert s2["decode_passes_per_s"] == pytest.approx(20.0) and s2["rework_passes"] == {"n": 6, "share": 0.2308}
    assert s2["decode_tokens_per_s"] == pytest.approx(18 / 1.3, abs=1e-2)
    # rows of earlier runs have no pass count: derived from steps, rollbacks and the stop reason
    old = {"steps": 12, "backtracks": 2, "stop_reason": "eos"}
    assert bench.passes_of(old) == 9 and bench.passes_of({**old, "stop_reason": "max_tokens"}) == 10
    assert bench.summarize([{k: v for k, v in _reply().items() if k not in ("decode_passes", "rework_passes")}])["rework_passes"] is None
    assert s["names_fused_class"]["share"] == 0.5 and s["names_fused_class"]["n"] == 2
    assert s["check_failures"]["exclamation"] == 1 and s["low_confidence_replies"] == 1


def _res(ft95, pps, ft50, n_tok=10.0, ok=1.0, p50=None):
    """a 4-thread summary: first-token p95 and p50, decode passes per second, mean tokens per reply."""
    return {"4": {"summary": {"first_token_ms": {"p95": ft95, "p50": ft50}, "decode_passes_per_s": pps,
                              "tokens_per_reply": {"mean": n_tok}, "full_reply_ms": {"p50": p50}, "check_pass_rate": ok}}}


def test_choice_rule(bench):
    small, big = "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"
    # the largest model that meets both targets; int8 unless q4 is more than 20 % faster on a reply of the same
    # length (first-token p50 + n_ref passes, n_ref the pooled mean length) with no lower pass rate
    r = {f"{small}/model_int8.onnx": _res(200, 30, 80), f"{small}/model_q4.onnx": _res(150, 40, 60),
         f"{big}/model_int8.onnx": _res(340, 16, 150), f"{big}/model_q4.onnx": _res(300, 20, 100)}
    c = bench.choose(r)
    assert c["deployed"] == f"{big}/model_q4.onnx" and c["meets_targets"]           # 775 / 600 ms = 1.29x faster
    r[f"{big}/model_q4.onnx"] = _res(300, 20, 100, ok=0.9)                            # ... but a lower pass rate
    assert bench.choose(r)["deployed"] == f"{big}/model_int8.onnx"
    r[f"{big}/model_q4.onnx"] = _res(300, 18, 130)                                   # 775 / 686 ms = 1.13x: int8 stays
    assert bench.choose(r)["deployed"] == f"{big}/model_int8.onnx"
    r[f"{big}/model_int8.onnx"] = _res(360, 16, 150)                                 # int8 misses the first-token target
    assert bench.choose(r)["deployed"] == f"{big}/model_q4.onnx"
    r[f"{big}/model_q4.onnx"] = _res(300, 14, 130)                                   # and q4 the decode rate: the small model
    assert bench.choose(r)["deployed"] == f"{small}/model_q4.onnx"                   # 413 / 310 ms = 1.33x
    # a file that writes longer replies is not counted as slower: q4's full-reply p50 is the larger one,
    # but per token it is twice as fast (n_ref 14: 800 vs 440 ms)
    r = {f"{small}/model_int8.onnx": _res(200, 20, 100, n_tok=8, p50=500), f"{small}/model_q4.onnx": _res(200, 40, 90, n_tok=20, p50=590)}
    c = bench.choose(r)
    assert c["deployed"] == f"{small}/model_q4.onnx" and "14.00 tokens" in c["reason"]
    assert bench.reply_cost_ms(c["candidates"][0], 14) == pytest.approx(800.0)
    r = {f"{small}/model_int8.onnx": _res(400, 30, 80), f"{small}/model_q4.onnx": _res(380, 10, 60),
         f"{big}/model_int8.onnx": _res(900, 5, 300)}
    c = bench.choose(r)
    assert c["deployed"] == f"{small}/model_q4.onnx" and c["meets_targets"] is False
    assert c["edit"] == 'DEFAULT_MODEL = "Qwen2.5-0.5B-Instruct", DEFAULT_ONNX = "model_q4.onnx" in meld_rt/response/generator.py'


def test_make_state_follows_the_service_rules(bench):
    emo9, ev11 = sv9(unknown=0.6, anger=0.3, neutral=0.1), [0.0] * 11
    face = {"p_face": p7(anger=0.5), "p_face_raw": p7(joy=0.6), "track": 2, "n_frames": 30, "presence": 0.9, "n_tracks": 1,
            "asd_corr": 0.4}
    d = bench.make_state(5, "you did what", 0.8, [], p7(anger=0.7), emo9, ev11, p7(anger=0.65), "stacked_tav", 0.2, face, 0.1, "asd")
    s = State.from_dict(d)
    assert s.vision_mode and s.vision.present and s.vision.rule == "asd" and s.vision.track == 2
    assert s.fused.label == "anger" and s.fused.confidence == 0.65 and s.fused.mode == "stacked_tav" and s.fused.calibrated
    assert s.discrepancy.face_flag and s.discrepancy.face_direction == "face_more_positive"
    assert "commit" in d and d["vision"]["emotion"]["anger"] == pytest.approx(0.5)
    g = bench.make_state(6, "", 0.0, ["asr_failed"], p7(anger=0.7), emo9, ev11, p7(anger=0.65), "stacked", 0.2, None, None)
    s = State.from_dict(g)
    assert s.fused.label == "neutral" and s.fused.confidence == 0.0 and s.degraded == ["asr_failed", "no_face"]
    assert not s.vision.present and s.discrepancy.flag is False
    assert G.build_messages(g, LOW, HIGH)[2] == []


@pytest.fixture(scope="module")
def vision_ws(tmp_path_factory):
    """package A's synthetic workspace (tests/test_train_vision.py): csv subsets, caches,
    the two-way heads and train_vision.py's heads, manifest and cache/vision_post_*.npz."""
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    import test_train_vision as tv

    return tv.ws.__wrapped__(tmp_path_factory)


def test_bench_states_synthetic_and_vision(bench, vision_ws):
    ws = vision_ws
    for synthetic in (True, False):
        rows, info = bench.build_states("test", 10, 0, synthetic, ws["cache"], ws["out"], ws["meld"])
        assert info["n_gold"] == 5 and info["n_live"] == 5 and len(rows) == 10
        assert [r["variant"] for r in rows] == ["gold"] * 5 + ["live"] * 5
        assert len({r["key"] for r in rows}) == 10
        assert info["source"] == ("synthetic" if synthetic else "cache/vision_post_test.npz")
        for r in rows:
            s = State.from_dict(r["state"])
            assert s.vision_mode and s.fused.mode == ("stacked" if synthetic else "stacked_tav")
            if s.degraded and s.degraded[0] in ("asr_failed", "low_asr_conf"):
                assert r["variant"] == "live" and s.fused.confidence == 0.0
            msgs, grounding, _ = G.build_messages(r["state"], *info["thresholds"])
            assert msgs[-1]["role"] == "user" and grounding
        if synthetic:
            assert not any(r["state"]["vision"]["present"] for r in rows)
        else:
            post = np.load(ws["cache"] / "vision_post_test.npz")
            present = dict(zip(post["keys"].tolist(), post["present"].tolist()))
            assert [r["state"]["vision"]["present"] for r in rows] == [bool(present[r["key"]]) for r in rows]
            assert info["thresholds"] == bench.thresholds(ws["manifest"], True)
    dev, info = bench.build_states("dev", 6, 0, True, ws["cache"], ws["out"], ws["meld"])
    assert info["n_live"] == 0 and {r["variant"] for r in dev} == {"gold"}


def test_bench_main_with_a_tiny_model(bench, vision_ws, tmp_path, monkeypatch):
    ws = vision_ws
    llm = tmp_path / "llm"
    d = _fake_lm(llm, ROWS)
    monkeypatch.setattr(G, "LLM_ROOT", d.parent)
    out = tmp_path / "bench.json"
    assert bench.main(["--synthetic", "6", "--models", "fake-llm/model_int8.onnx", "--threads", "1", "2", "--cache", str(ws["cache"]),
                       "--heads", str(ws["out"]), "--meld", str(ws["meld"]), "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    res = r["results"]["fake-llm/model_int8.onnx"]
    assert set(res) == {"1", "2", "same_text_across_threads"} and res["same_text_across_threads"] == 1.0
    s = res["2"]["summary"]
    assert s["n"] == 6 and s["first_token_ms"]["p50"] > 0 and 0 <= s["check_pass_rate"] <= 1
    assert len(r["examples"]) == 6 and {"key", "evidence", "grounding", "allowed", "text", "checks"} <= set(r["examples"][0])
    assert r["choice"]["reason"] == "no 4-thread results" and r["states"]["source"] == "synthetic"


# ---------------------------------------------------------------- fetch_models groups and provenance rows

def test_fetch_models_default_is_core_and_keeps_other_rows(tmp_path, monkeypatch):
    import hashlib
    import sys

    sys.path.insert(0, sys_path_tools)
    import fetch_models as fm

    assert fm.DEFAULT_GROUPS == "core"
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "a.bin").write_bytes(b"core file")
    sha = hashlib.sha256(b"core file").hexdigest()
    table = [fm.Asset("a", "http://127.0.0.1:9/a", "a.bin", sha256=sha),
             fm.Asset("b", "http://127.0.0.1:9/b", "vision/b.onnx", sha256="0" * 64, group="vision"),
             fm.Asset("c", "http://127.0.0.1:9/c", "llm/c.onnx", sha256="1" * 64, group="llm")]
    monkeypatch.setattr(fm, "TABLE", table)
    prov = tmp_path / "prov.json"
    earlier = [{"asset": "a", "path": "old", "ok": False}, {"asset": "b", "path": "models/vision/b.onnx", "ok": True, "group": "vision"},
               {"asset": "c", "path": "models/llm/c.onnx", "ok": True, "group": "llm"}, {"asset": "gone", "ok": True}]
    prov.write_text(json.dumps({"files": earlier}))
    args = ["--dest", str(dest), "--dl-dir", str(tmp_path / "dl"), "--provenance", str(prov), "--check"]
    # the default checks the core group only: its row is rewritten, the vision and llm rows are kept as they were
    assert fm.main(args) == 0
    out = json.loads(prov.read_text())
    assert [r["asset"] for r in out["files"]] == ["a", "b", "c"] and out["all_ok"] is True
    assert out["files"][0]["sha256"] == sha and out["files"][1:] == earlier[1:3]
    # a group that is checked and missing makes the exit status and all_ok false; the unchecked row stays
    assert fm.main([*args, "--groups", "core,vision"]) == 1
    out = json.loads(prov.read_text())
    assert [r["asset"] for r in out["files"]] == ["a", "b", "c"] and out["all_ok"] is False
    assert out["files"][1]["exists"] is False and out["files"][2] == earlier[2]
    # no earlier file: only the checked rows
    prov.unlink()
    assert fm.main(args) == 0 and [r["asset"] for r in json.loads(prov.read_text())["files"]] == ["a"]


# ---------------------------------------------------------------- parameter count of a quantized LLM

def test_count_onnx_weights_counts_each_weight_once(tmp_path):
    import sys

    import onnx
    from onnx import TensorProto, helper

    sys.path.insert(0, sys_path_tools)
    import count_params as cp

    def t(name, dt, dims):
        n = int(np.prod(dims)) if dims else 1
        vals = np.zeros(n, dtype=np.int64 if dt == TensorProto.INT64 else np.float32)
        if dt in (TensorProto.INT8, TensorProto.UINT8):
            return helper.make_tensor(name, dt, dims, vals.astype(np.int8 if dt == TensorProto.INT8 else np.uint8).tobytes(), raw=True)
        return helper.make_tensor(name, dt, dims, vals.tolist())

    inits = [t("embed_quantized", TensorProto.UINT8, [10, 4]), t("embed_scale", TensorProto.FLOAT, []),
             t("embed_zero_point", TensorProto.UINT8, []),
             t("w1_quantized", TensorProto.INT8, [4, 6]), t("w1_zero_point", TensorProto.INT8, []), t("w1_scale", TensorProto.FLOAT, []),
             t("w2_Q4", TensorProto.UINT8, [4, 1, 3]), t("w2_scales", TensorProto.FLOAT, [4]),
             t("norm.weight", TensorProto.FLOAT, [4]), t("q.bias", TensorProto.FLOAT, [6]),
             t("/model/rotary/Constant_output_0", TensorProto.FLOAT, [32, 2]), t("/model/Constant_1", TensorProto.FLOAT, []),
             t("shape_const", TensorProto.INT64, [3])]
    nodes = [helper.make_node("DequantizeLinear", ["embed_quantized", "embed_scale", "embed_zero_point"], ["e"]),
             helper.make_node("MatMulInteger", ["xq", "w1_quantized", "xz", "w1_zero_point"], ["y1"]),
             helper.make_node("MatMulNBits", ["x", "w2_Q4", "w2_scales"], ["y2"], domain="com.microsoft", K=6, N=4, bits=4, block_size=32)]
    g = helper.make_graph(nodes, "q", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6])],
                          [helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 4])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)])
    p = tmp_path / "q.onnx"
    onnx.save(m, str(p))
    c = cp.count_onnx_weights(p)
    # embedding 40 + int8 matmul 24 + 4-bit matmul K x N 24 (not its 12 packed bytes) + norm 4 + bias 6
    assert c["params"] == 40 + 24 + 24 + 4 + 6
    assert c["by_dtype"] == {"UINT8": 40, "INT8": 24, "UINT8_4bit": 24, "FLOAT": 10}
    assert c["excluded"] == {"scales_zero_points": 1 + 1 + 1 + 1 + 4, "exporter_constants": 64, "scalar_constants": 1,
                             "integer_constants": 3}


def test_count_params_extension_table(tmp_path):
    import sys

    sys.path.insert(0, sys_path_tools)
    import count_params as cp

    models, heads_dir = tmp_path / "models", tmp_path / "heads"
    heads_dir.mkdir()
    fake = _fake_lm(tmp_path / "src", ROWS)
    dst = models / "llm" / G.DEFAULT_MODEL / "onnx"
    dst.mkdir(parents=True)
    (dst / G.DEFAULT_ONNX).write_bytes((fake / "onnx" / "model_int8.onnx").read_bytes())
    n = 15
    np.savez(heads_dir / "commit_policy.npz", feature_names=np.array([f"f{i}" for i in range(n)]), mean=np.zeros(n), scale=np.ones(n),
             p_w=np.zeros(n), p_b=np.array(0.0), c=np.array(0.05), qw_kind=np.array("mlp"), qw_W1=np.zeros((n, 16)),
             qw_b1=np.zeros(16), qw_W2=np.zeros(16), qw_b2=np.array(0.0))
    res = cp.build(models, heads_dir)
    by = {r["component"]: r for r in res["extension_rows"]}
    v = len(_bpe_tokenizer().get_vocab(with_added_tokens=True)) + 3
    llm = by[f"{G.DEFAULT_MODEL} ({G.DEFAULT_ONNX})"]
    assert llm["params"] == 2 * v * v + 2 * v and llm["excluded"] == {"integer_constants": 1}
    assert by["CommitPolicy"]["params"] == n + 1 + n * 16 + 16 + 16 + 1
    assert by["YuNet 2023mar"]["params"] is None and by["Fusion3"]["note"] == "pending"
    assert res["total_extension_only"] == llm["params"] + by["CommitPolicy"]["params"]
    assert res["total_with_extension"] == res["total_inference_path"] + res["total_extension_only"]
    res["generated_at"] = "test"
    md = cp.render(res, models, heads_dir)
    assert "Extension (`--vision --reply llm`, spec 4)" in md and "**Total with the extension**" in md
