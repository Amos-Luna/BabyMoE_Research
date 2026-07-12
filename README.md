# BabyMoE Research

Training and analysis pipeline for **BabyMoE: Language-Specific Expert Routing
in Small Multilingual Language Models**, targeting the BabyLM 2026 Multilingual
Track (English, Dutch, and Chinese).

The submitted models are initialized from scratch. The MoE condition uses a
native Hugging Face Mixtral architecture with four experts and top-2 routing;
the dense Mistral control is matched for active FFN parameters and FLOPs.

## Colab workflow

1. Open `notebooks/01_train_colab.ipynb`.
2. Set the GitHub URL and persistent Google Drive root.
3. Accept the three gated BabyLM dataset terms, then add `HF_TOKEN` (and
   optionally `WANDB_API_KEY`) to Colab Secrets.
4. Run the cells to bootstrap, prepare data, and train.
5. Open `notebooks/02_upload_hf.ipynb` to audit and publish genuine
   `chck_{N}M` revisions.

All large downloads, prepared blocks, tokenizer files, logs, resume states, and
checkpoints live under the configured Drive root. Re-running preparation skips
artifacts whose manifests and hashes are valid.

## Local entry points

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py --config configs/moe_learned.yaml
python scripts/train.py --config configs/moe_learned.yaml --seed 42 \
  --stop-at-exposure 10000000
python scripts/analyze_routing.py --run-dir artifacts/runs/<run>
python scripts/upload_to_hf.py --run-dir artifacts/runs/<run> \
  --repo-id USER/BabyMoE --dry-run
```

See `docs/02_experiment_plan.md` for the four-day experiment protocol.
