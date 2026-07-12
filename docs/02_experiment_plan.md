# BabyMoE four-day experiment protocol

## Research claim and tests

BabyMoE asks whether a small multilingual MoE learns a layerwise division of
labor that is not reducible to token identity, writing system, frequency,
position, or corpus domain.

- **H1 — residual routing information.** Adding router probabilities to a
  document-grouped control probe should improve three-way language macro-AUROC.
  The preregistered per-layer signal is a gain of at least 0.03, a
  document-bootstrap 95% interval above zero, and a within-stratum permutation
  \(p<0.05\).
- **H2 — depth profile.** English–Chinese routing divergence should be larger
  at the network edges than in the middle:
  \(U=\operatorname{mean}(JS_{\mathrm{edge}})-
  \operatorname{mean}(JS_{\mathrm{middle}})>0\). English–Dutch is expected to
  be the most similar pair, but this is descriptive rather than a gate.
- **H3 — matched predictive quality.** At 100M exposure, learned MoE validation
  cross-entropy should be no more than 5% worse than the dense control while
  the router remains non-collapsed.

## Locked compliance and compute

Only `BabyLM-community/babylm-{eng,nld,zho}` training text is used. For language
\(\ell\), competition exposure is
\(u_\ell=\mathrm{raw\_count}_\ell/p_\ell\), using configurable
CfP premiums `{eng: 1.0, nld: 1.0516, zho: 0.9894}`. The Chinese dataset card
currently reports `0.935966`; every manifest records the chosen source and
value so the corpus can be rebuilt if organizers clarify it.

The selected equal-share corpus is capped at 100M adjusted units. Validation
documents are disjoint. Repeated exposure is capped at ten passes and 1B
adjusted units. Model subword tokens are logged separately and are never used
as competition exposure.

Both models have 12 layers, width 768, 12 attention heads, context 1024, and the
same 16,384-entry joint byte-level BPE tokenizer. Its training sample is a
deterministic 10M-character cap per language drawn only from selected training
documents. Dense SwiGLU width is 2048;
MoE has four 1024-wide experts with top-2 routing. Active FFN parameters match:

\[
3(768)(2048)=2\cdot3(768)(1024)=4{,}718{,}592
\]

per layer. Total parameters are intentionally larger for MoE and are reported
alongside the active-parameter/FLOP proxy.

## Experiment sequence

1. **Milestone 0 — smoke.** Prepare synthetic permitted-format data and run the
   tiny `smoke.yaml` model through forward, backward, resume, checkpoint,
   routing aggregation, and local Hugging Face reload.
2. **Milestone 1 — 10M pilot.** Train learned MoE and dense controls with paired
   seeds 13, 42, and 71. Train one frozen-random-gate MoE with seed 42. Evaluate
   validation CE/perplexity by language and analyze every available routing
   snapshot.
3. **Milestone 2 — 100M.** Resume seed 42 for learned MoE and dense from exactly
   saved trainer state. Do not restart either condition or change data order.
4. **Milestone 3 — optional continuation.** Continue both seed-42 runs
   symmetrically toward 1B only if the A100 benchmark leaves enough wall time,
   manifests remain valid, and learned MoE is within 5% of dense CE at 100M.

The frozen-gate control distinguishes learned specialization from differences
created by random expert partitioning. It retains the same load-balancing
coefficient as learned MoE, freezes only gate parameters, and is not continued
beyond 10M.
Pilot stops are prefixes of the 1B trajectory: WSD is never rescaled to a
shorter CLI stop, so the first 10M units are the same warmup that will be
resumed for the 100M and optional 1B runs.

## Measurements and decision rules

Training logs contain CE, total and load-balancing loss, learning rate, gradient
norm, model-token throughput, memory, ETA, global/per-language adjusted
exposure, and language epochs. Checkpoint validation records CE and perplexity
for all three languages.

Routing artifacts contain per layer and language: mean expert probabilities,
top-2 load, normalized entropy, load coefficient of variation, minimum expert
share, pairwise Jensen–Shannon divergence, and stability from the prior
checkpoint. Compact token snapshots retain token ID, position, document group,
domain, and router probabilities.

The control probe uses token ID, Unicode script, log-frequency bin, position
bin, and domain. The augmented probe adds one layer's router probabilities.
Splits are grouped by document. Confidence intervals bootstrap test documents;
permutation shuffles labels within script/frequency/position strata.

- A positive 10M mechanism requires residual routing information and \(U>0\)
  in at least two of three seeds.
- A healthy null (no residual signal in all seeds, no collapse) still proceeds
  to 100M because scale is part of the research question.
- A router is operationally healthy when minimum top-2 expert share is at least
  0.05 and maximum load CV is at most 1.0.
- Mixed or unstable pilot results trigger diagnosis, not an unplanned
  architecture search. No additional variants are added during the four days.

## Four-day execution

- **Day 1:** Colab bootstrap, synthetic smoke, freeze data/tokenizer manifests,
  benchmark both full architectures, then launch paired 10M pilots.
- **Day 2:** finish pilots, run controlled probes, frozen-gate control, and
  `summarize_pilots.py`; the notebook automatically resumes seed 42 conditions
  to 100M if the pilot gate passes.
- **Day 3:** finish 100M, publish genuine revisions, run official zero-shot
  learning-curve evaluation, and continue both runs only if the measured ETA
  permits.
- **Day 4:** run full official evaluation on final learned-MoE and dense models,
  collate submissions, freeze analysis JSON, figures, manifests, and paper
  results.

## Artifacts and recovery

Google Drive stores the Hugging Face cache, selected Parquet documents,
tokenizer, packed memory-mapped blocks, manifests, run logs, rotating trainer
state, checkpoints, routing snapshots, and upload state. Stage fingerprints and
SHA-256 records make preparation and upload idempotent. Resume is refused if
the resolved config, corpus manifest, or tokenizer manifest changed.
The notebook copies only packed blocks to Colab SSD for A100 throughput; its
tokenizer, run directory, resume state, and checkpoints remain linked to Drive.
On reconnect it checks out the Git SHA recorded by persistent runs instead of
pulling newer code, preserving exact optimizer/data/RNG resume semantics.

Inference checkpoints are created only after crossing a real exposure
milestone: 1M–10M by 1M, 20M–100M by 10M, and 200M–1B by 100M. The Hub `main`
branch points to the true final state; no missing revision is fabricated.

## Official evaluation handoff

The models use native `MistralForCausalLM`/`MixtralForCausalLM`, and the final
checkpoint is audited through causal-, sequence-, and token-classification
AutoModel interfaces. Run the current official scripts from
`babylm-eval/multilingual/`:

```bash
# Final model: zero-shot, Global PIQA, MECO, and fine-tuning.
bash scripts/eval_model_full.sh --model_name ORG/MODEL --bos_fix 0

# One genuine pilot revision when later revisions do not yet exist.
bash scripts/zeroshot_model.sh --model_name ORG/MODEL \
  --revision chck_10M --bos_fix 0

# Full learning curve only after all revisions expected by the wrapper exist.
bash scripts/eval_model_fast.sh --model_name ORG/MODEL --bos_fix 0

python scripts/collate_results.py --model_name ORG/MODEL --fast
```

`--bos_fix 0` is intentional: packed training inserts EOS document boundaries
but never prepends BOS. Full official evaluation is a separate execution phase;
the training pipeline produces evaluator-compatible models and revisions.
