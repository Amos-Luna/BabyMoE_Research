#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_ROOT="/content/drive/MyDrive/BabyMoE"
SKIP_INSTALL=0
SKIP_GPU_CHECK=0
RUN_SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drive-root) DRIVE_ROOT="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    --run-smoke) RUN_SMOKE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$(dirname "$DRIVE_ROOT")" ]]; then
  echo "Google Drive is not mounted: $(dirname "$DRIVE_ROOT")" >&2
  exit 1
fi

mkdir -p "$DRIVE_ROOT/artifacts"/{data,tokenizer,runs,hf_cache,smoke}

if [[ -e "$REPO_ROOT/artifacts" && ! -L "$REPO_ROOT/artifacts" ]]; then
  cp -a -n "$REPO_ROOT/artifacts/." "$DRIVE_ROOT/artifacts/" 2>/dev/null || true
  mv "$REPO_ROOT/artifacts" "$REPO_ROOT/artifacts.local-backup.$(date +%s)"
fi
ln -sfn "$DRIVE_ROOT/artifacts" "$REPO_ROOT/artifacts"

export HF_HOME="$DRIVE_ROOT/artifacts/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_DATASETS_CACHE"

if [[ "$SKIP_GPU_CHECK" -eq 0 ]]; then
  python - <<'PY'
import re
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required for the full pipeline")
match = re.match(r"(\d+)\.(\d+)", torch.__version__)
if match is None or tuple(map(int, match.groups())) < (2, 4):
    raise SystemExit(f"Colab PyTorch >=2.4 is required; found {torch.__version__}")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"[bootstrap] GPU={name} memory={memory:.1f} GiB torch={torch.__version__}")
if "A100" not in name:
    print("[bootstrap-warning] This configuration is tuned for an A100; run the smoke test before training.")
PY
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  TORCH_VERSION="$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])')"
  TORCH_CONSTRAINT="$(mktemp)"
  echo "torch==$TORCH_VERSION" > "$TORCH_CONSTRAINT"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet \
    --constraint "$TORCH_CONSTRAINT" \
    -r "$REPO_ROOT/requirements.txt"
  rm -f "$TORCH_CONSTRAINT"
fi

if [[ "$RUN_SMOKE" -eq 1 ]]; then
  python "$REPO_ROOT/scripts/prepare_data.py" \
    --config "$REPO_ROOT/configs/smoke.yaml" \
    --artifacts-root "$DRIVE_ROOT/artifacts" \
    --synthetic
  python "$REPO_ROOT/scripts/train.py" \
    --config "$REPO_ROOT/configs/smoke.yaml" \
    --artifacts-root "$DRIVE_ROOT/artifacts" \
    --seed 42
fi

echo "[bootstrap] persistent root: $DRIVE_ROOT/artifacts"
echo "[bootstrap] setup complete"
