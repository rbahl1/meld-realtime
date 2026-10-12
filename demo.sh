#!/usr/bin/env bash
# service + monitor + replay on one clip; prints the state and response lines
# and writes every message to results/demo_log.jsonl.
# usage: ./demo.sh [clip.wav]      (default proxy/dev_0001.wav, else en.wav)
# env: THREADS (4), ENDPOINT_MS (500), N_FINALS (1), DEMO_LOG (results/demo_log.jsonl)
set -euo pipefail
cd "$(dirname "$0")"

CLIP="${1:-proxy/dev_0001.wav}"
if [ ! -f "$CLIP" ]; then
  echo "no clip at $CLIP; using models/sensevoice/test_wavs/en.wav"
  CLIP=models/sensevoice/test_wavs/en.wav
fi
THREADS="${THREADS:-4}"
ENDPOINT_MS="${ENDPOINT_MS:-500}"
N_FINALS="${N_FINALS:-1}"
LOG="${DEMO_LOG:-results/demo_log.jsonl}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

cleanup() { kill "${SVC:-}" "${MON:-}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== service: threads=$THREADS endpoint=${ENDPOINT_MS}ms"
uv run python -m meld_rt.service --threads "$THREADS" --endpoint-silence-ms "$ENDPOINT_MS" --preload --log-level warning &
SVC=$!

echo "== monitor: state and response lines below, everything in $LOG"
uv run python -m meld_rt.clients.monitor --jsonl "$LOG" --topics state.provisional state.final response error \
  --exit-after-final "$N_FINALS" --idle-s 20 --max-s 180 &
MON=$!

echo "== replay: $CLIP at real-time pace (waits for the models to load)"
uv run python -m meld_rt.clients.replay --wav "$CLIP" --realtime --stream-id demo --wait-ready 180

wait "$MON" || true
echo "== summary"
uv run python - "$LOG" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
finals = [r for r in rows if r["topic"] == "state.final"]
replies = {r["utt_id"]: r for r in rows if r["topic"] == "response"}
print(f"{sum(r['topic'] == 'state.provisional' for r in rows)} provisional, {len(finals)} final, {len(replies)} response, "
      f"{sum(r['topic'] == 'error' for r in rows)} error")
for f in finals:
    tm, fu, d = f["timing_ms"], f["fused"], f["discrepancy"]
    print(f"utt {f['utt_id']} media {f['t_media_start']}-{f['t_media_end']} s")
    print(f"  transcript  {f['asr']['text']!r} (conf {f['asr']['conf']})")
    print(f"  audio       {max(f['audio']['emotion'], key=f['audio']['emotion'].get)}  text {max(f['text']['emotion'], key=f['text']['emotion'].get)}"
          f"  fused {fu['label']}@{fu['confidence']} ({fu['mode']}, calibrated={fu['calibrated']}, sentiment {fu['sentiment']})")
    print(f"  discrepancy jsd {d['jsd']} flag {d['flag']} direction {d['direction']}  degraded {f['degraded']}")
    print(f"  timing      service_total {tm['service_total']} ms (sensevoice {tm['sensevoice']}, text {tm['text_encoder']}, "
          f"prosody {tm['prosody']}, heads {tm['heads']}, provisional in flight {tm['provisional_in_flight']}); "
          f"client-side {f.get('latency_client_ms')} ms; vad tail {tm['vad_tail']} ms")
    r = replies.get(f["utt_id"])
    if r:
        print(f"  reply       {r['text']!r} [{r['template_id']}, {r['style']}] after_final {r.get('after_final_ms')} ms")
        print(f"  grounding   {r['grounding']}")
EOF
