#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier

fetch_asset() {
  python3 - "$1" "$2" "$3" <<'FETCH'
import hashlib, os, sys, urllib.request
url, sha, dest = sys.argv[1:4]
def ok():
    return (os.path.exists(dest) and
            hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha)
if not ok():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    assert ok(), f"asset checksum mismatch: {dest}"
FETCH
}

fetch_asset "https://openaievalsetub0ktk.blob.core.windows.net/assets/FreeCAD/4_sling_lift/tests/task/reference/solution.FCStd" "83c606e4e5642b89b84ed306b3f9cc54aa1377986d1c6292b4bbb464ae756399" "/tests/task/reference/solution.FCStd"

CANDIDATE=/app/solution.FCStd

if [ ! -f "$CANDIDATE" ]; then
  echo "no $CANDIDATE produced" > /logs/verifier/harness_log.txt
  echo '{"reward": 0.0, "score": 0, "max_score": 0, "passed": false}' \
    > /logs/verifier/reward.json
  exit 0
fi

python3 /tests/task/harness/harness.py "$CANDIDATE" \
  > /logs/verifier/harness_result.json 2> /logs/verifier/harness_log.txt

python3 - <<'PY'
import json

# Harbor's reward.json schema requires every value to be a plain number
# (or bool) -- no nested objects or strings. The full breakdown (subscores,
# task_id, ...) stays in harness_result.json alongside this file.
try:
    with open("/logs/verifier/harness_result.json") as f:
        result = json.load(f)
    max_score = result.get("max_score") or 0
    score = result.get("score") or 0
    reward = (score / max_score) if max_score else 0.0
    payload = {"reward": reward, "score": score, "max_score": max_score,
               "passed": bool(result.get("passed"))}
except Exception:
    payload = {"reward": 0.0, "score": 0, "max_score": 0, "passed": False}

with open("/logs/verifier/reward.json", "w") as f:
    json.dump(payload, f, indent=2)
PY
