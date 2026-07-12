#!/usr/bin/env bash
set -euo pipefail

PERSISTENT_ARTIFACTS="${1:?usage: stage_for_training.sh PERSISTENT_ARTIFACTS LOCAL_ARTIFACTS}"
LOCAL_ARTIFACTS="${2:?usage: stage_for_training.sh PERSISTENT_ARTIFACTS LOCAL_ARTIFACTS}"

if [[ "$LOCAL_ARTIFACTS" == "/" || "$LOCAL_ARTIFACTS" == "$PERSISTENT_ARTIFACTS" ]]; then
  echo "Local staging root must be distinct and may not be /" >&2
  exit 2
fi

if [[ ! -f "$PERSISTENT_ARTIFACTS/data/blocks/manifest.json" ]]; then
  echo "Prepared block manifest not found under $PERSISTENT_ARTIFACTS" >&2
  exit 1
fi

mkdir -p "$LOCAL_ARTIFACTS/data"
rm -rf "$LOCAL_ARTIFACTS/data/blocks"
cp -a "$PERSISTENT_ARTIFACTS/data/blocks" "$LOCAL_ARTIFACTS/data/blocks"
python - "$LOCAL_ARTIFACTS/data/blocks" <<'PY'
import json
import sys
from pathlib import Path

from babymoe.utils import manifest_is_valid

root = Path(sys.argv[1])
manifest = root / "manifest.json"
payload = json.loads(manifest.read_text(encoding="utf-8"))
if not manifest_is_valid(
    manifest,
    expected_fingerprint=payload["fingerprint"],
    root=root,
):
    raise SystemExit("local block staging failed manifest/hash verification")
PY

for directory in tokenizer runs hf_cache; do
  rm -rf "$LOCAL_ARTIFACTS/$directory"
  ln -s "$PERSISTENT_ARTIFACTS/$directory" "$LOCAL_ARTIFACTS/$directory"
done

echo "[stage] local blocks: $LOCAL_ARTIFACTS/data/blocks"
echo "[stage] persistent tokenizer/runs/cache: $PERSISTENT_ARTIFACTS"
