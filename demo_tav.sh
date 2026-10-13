#!/usr/bin/env bash
# three inputs: the service with --vision --reply llm, a monitor and the replay
# client streaming one mp4 (audio and frames) at real-time pace; prints a summary
# per utterance and writes every message to results/demo_tav_log.jsonl.
# usage: ./demo_tav.sh [clip.mp4]      (default: $DEMO_MP4)
# env: DEMO_MP4, THREADS (4), ENDPOINT_MS (500), N_FINALS (1), VIDEO_FPS (10),
#      REPLY (llm), LLM_DIR (default: the only model folder under models/llm/),
#      DEMO_LOG (results/demo_tav_log.jsonl)
set -euo pipefail
cd "$(dirname "$0")"

CLIP="${1:-${DEMO_MP4:-}}"
if [ -z "$CLIP" ] || [ ! -f "$CLIP" ]; then
  echo "no mp4 at '${CLIP}': give one as the first argument or set DEMO_MP4" >&2
  exit 2
fi
THREADS="${THREADS:-4}"
ENDPOINT_MS="${ENDPOINT_MS:-500}"
N_FINALS="${N_FINALS:-1}"
VIDEO_FPS="${VIDEO_FPS:-10}"
REPLY="${REPLY:-llm}"
LOG="${DEMO_LOG:-results/demo_tav_log.jsonl}"
LLM_ARGS=()
if [ -n "${LLM_DIR:-}" ]; then LLM_ARGS=(--llm-dir "$LLM_DIR"); fi
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

cleanup() { kill "${SVC:-}" "${MON:-}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== service: --vision --reply $REPLY threads=$THREADS endpoint=${ENDPOINT_MS}ms"
uv run python -m meld_rt.service --vision --reply "$REPLY" ${LLM_ARGS[@]+"${LLM_ARGS[@]}"} --threads "$THREADS" \
  --endpoint-silence-ms "$ENDPOINT_MS" --preload --log-level warning &
SVC=$!

echo "== monitor: state, commit and reply lines below, everything in $LOG"
uv run python -m meld_rt.clients.monitor --jsonl "$LOG" \
  --topics state.provisional state.final state.commit response.delta response error \
  --exit-after-final "$N_FINALS" --idle-s 30 --max-s 300 &
MON=$!

echo "== replay: $CLIP (audio and $VIDEO_FPS fps video) at real-time pace (waits for the models to load)"
uv run python -m meld_rt.clients.replay --video "$CLIP" --video-fps "$VIDEO_FPS" --realtime --stream-id demo --wait-ready 300

wait "$MON" || true
echo "== summary"
uv run python - "$LOG" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
top = lambda d: max(d, key=d.get) if d else "-"
finals = [r for r in rows if r["topic"] == "state.final"]
replies = {r["utt_id"]: r for r in rows if r["topic"] == "response"}
commits = {r["utt_id"]: r for r in rows if r["topic"] == "state.commit"}
deltas = {}
for r in rows:
    if r["topic"] == "response.delta":
        deltas.setdefault(r["utt_id"], []).append(r)
n = lambda t: sum(r["topic"] == t for r in rows)
print(f"{n('state.provisional')} provisional, {len(finals)} final, {n('state.commit')} commit, {n('response.delta')} delta, "
      f"{len(replies)} response, {n('error')} error")
for f in finals:
    tm, fu, d, v, c = f["timing_ms"], f["fused"], f["discrepancy"], f.get("vision"), f.get("commit")
    print(f"utt {f['utt_id']} media {f['t_media_start']}-{f['t_media_end']} s")
    print(f"  transcript  {f['asr']['text']!r} (conf {f['asr']['conf']})")
    face = (f"face {top(v['emotion'])}@{v['emotion'][top(v['emotion'])]:.2f} (raw {top(v['raw'])}, presence {v['presence']}, "
            f"track {v['track']} of {v['n_tracks']}, rule {v['rule']})" if v and v["present"] else "face none")
    print(f"  labels      text {top(f['text']['emotion'])}  voice {top(f['audio']['emotion'])}  {face}")
    print(f"  fused       {fu['label']}@{fu['confidence']} ({fu['mode']}, calibrated={fu['calibrated']}, sentiment {fu['sentiment']})")
    print(f"  discrepancy words-voice jsd {d['jsd']} flag {d['flag']} {d['direction']}; words-face jsd {d.get('face_jsd')} "
          f"flag {d.get('face_flag')} {d.get('face_direction')}  degraded {f['degraded']}")
    if c:
        before = round(f["t_media_end"] - f["t_media_start"] + tm["vad_tail"] / 1000 - c["t_commit_s"], 3)
        sc = commits.get(f["utt_id"])
        print(f"  commit      {c['label']}@{c['confidence']} at {c['t_commit_s']} s after the segment start ({c['source']}, "
              f"{before} s before the final, matches final {c['matches_final']})"
              + (f", q_emit {sc['q_emit']} q_wait {sc['q_wait']}" if sc else ""))
    else:
        print("  commit      none (no commit policy)")
    print(f"  timing      service_total {tm['service_total']} ms (sensevoice {tm['sensevoice']}, text {tm['text_encoder']}, "
          f"vision wait {tm.get('vision_wait')}, vision aggregate {tm.get('vision_aggregate')}, heads {tm['heads']}, "
          f"provisional in flight {tm['provisional_in_flight']}); client-side {f.get('latency_client_ms')} ms; vad tail {tm['vad_tail']} ms")
    r = replies.get(f["utt_id"])
    if r:
        ds = deltas.get(f["utt_id"], [])
        print(f"  reply       {r['text']!r} [{r.get('source', 'bank')}, {r['template_id']}, fallback {r.get('fallback')}]")
        print(f"  streamed    {len(ds)} deltas, first {r.get('first_delta_ms')} ms after the endpoint frame arrived, "
              f"generation {r.get('gen_ms')} ms, {r.get('n_tokens')} tokens, checks {r.get('checks')}")
        print(f"  grounding   {r['grounding']}")
EOF
