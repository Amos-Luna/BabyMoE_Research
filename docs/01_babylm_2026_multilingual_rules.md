# BabyLM 2026 Multilingual: Rules and Considerations

**Last verified:** July 12, 2026.  
**Scope:** Rules for a competitive Multilingual submission, not merely a non-competitive workshop paper.

## Executive Summary

- **An NVIDIA A100 is not required.** BabyLM 2026 does not impose restrictions on accelerator type or FLOP budget. An A100 is an infrastructure choice, not a competition rule.
- The track focuses on **English, Dutch, and Chinese** and allows a custom language mixture equivalent to a **100M training budget under the official Byte Premium accounting**.
- For competitive ranking, a model may accumulate at most **1B equivalent exposure units**, typically corresponding to ten passes over a 100M-budget corpus. Repeated exposures count toward the limit.
- Architecture is unrestricted, but the submitted model must be able to assign **log-likelihood or pseudo-log-likelihood to text**, and the submission must be usable through Hugging Face / Transformers.
- Intermediate checkpoints and official zero-shot and fine-tuning evaluation are required.
- Initializing from a GPT-2, Qwen, or any checkpoint trained on external corpora, or using a tokenizer trained on external linguistic data, **should not be assumed compliant unless explicitly confirmed by the organizers**. The budget includes linguistic information seen by learned tools and auxiliary models.
- The public timeline currently lists **July 15, 2026** as the direct submission deadline, while some references may differ. The final submission date should be verified immediately through OpenReview and the official communication channels.

## 1. Track, Languages, and Data

The new Multilingual track uses the infrastructure of [BabyBabelLM](https://aclanthology.org/2026.eacl-long.152/) and evaluates:

- English (`eng` / `en`);
- Dutch (`nld` / `nl`);
- Chinese (`zho` / `zh`).

The explicit motivation is to compare a typologically close language pair with shared writing systems (English–Dutch) against a typologically and graphically distant language (Chinese).

### Budget

The published rules allow participants to create a mixture of the three languages totaling **100M training units adjusted through Byte Premium accounting**. The Call for Papers also specifies the competitive exposure cap:

- up to 100M first-exposure budget;
- up to 1B total exposure for Multilingual, including repetitions;
- in practice, at most ten epochs over a 100M-budget corpus;
- fewer exposures may also be used.

Published Byte Premium values relative to English are:

- English: `1.0000`;
- Dutch: `1.0516`;
- Chinese: `0.9894`.

Byte Premium measures the ratio of UTF-8 bytes required to express parallel content. For example, 100 MB of English content corresponds approximately to 105.16 MB in Dutch and 98.94 MB in Chinese. This is not equivalent to the number of tokenizer-generated subwords.

**Project implication:** allocating `3.33M raw tokens` per language does not automatically apply Byte Premium. An auditable accounting mechanism for equivalent words/bytes should be maintained using the official scripts or manifests. Since official documentation alternates between terms such as "tokens", "words", and "whitespace-separated words", the safest approach is to use the official accounting implementation and confirm the exact formula with the organizers before the final run.

### Allowed Sources and Traceability

The documentation allows participants to choose language proportions and replace data with custom sources while respecting the budget. For Multilingual submissions, the customized corpus mixture must be provided. The lowest-risk option is to use only the official BabyBabelLM splits for all three languages.

If a custom corpus is used:

- it must respect the budget and cognitively plausible data principles;
- provenance, licensing, filtering, deduplication, and language/domain counts should be preserved;
- the required corpus manifest and datasheet should be released;
- evaluation examples must never enter pretraining.

### Tokenizers, Taggers, and Auxiliary Models

The FAQ indicates that if a tool learned from language data, the linguistic data seen by that tool counts toward the budget. This potentially affects:

- pretrained tokenizers;
- external POS taggers or parsers used to generate training features;
- translation models;
- rerankers;
- synthetic data generators;
- models used for distillation.

Therefore, for a minimal-risk submission:

1. train a joint tokenizer exclusively on the permitted corpus, or use an officially documented compatible tokenizer;
2. avoid initializing from `gpt2`, `Qwen/Qwen3-0.6B`, or any checkpoint trained on external corpora unless explicit organizer guidance confirms compliance;
3. avoid using external multilingual embeddings during training;
4. reserve external tools for post-hoc analysis, making clear that they do not affect submitted model weights.

The Call for Papers explicitly discusses external models in Strict and Strict-Small settings, but does not provide equally explicit guidance for Multilingual. Therefore, teacher-model supervision or external learned models should not be assumed permissible without written organizer confirmation.

## 2. Architecture and Training Constraints

There is no mandatory architecture. Dense Transformers, MoE models, RNNs, State-Space Models, causal objectives, masked objectives, and hybrid objectives are all allowed, provided that:

- the submission can score strings using log-likelihood or pseudo-log-likelihood without prior fine-tuning;
- it integrates reproducibly with Hugging Face / Transformers;
- all pretraining exposure remains within the allowed budget;
- architecture, hyperparameters, data, and compute are reported.

There is no:

- requirement to use GPT-2;
- requirement to use the official baseline;
- requirement to use an A100;
- published parameter-count limit;
- published FLOP limit;
- optimizer or learning-rate restriction.

There is, however, a strict exposure/epoch limit. Training many experimental variants during development does not change the limit for an individual submitted model, although approximate compute usage should be reported and hidden evaluation tasks should not be used for model selection.

### Intermediate Checkpoints

Submissions must retain checkpoints according to equivalent exposure counts:

- every 1M until 10M: `1M, 2M, ..., 10M`;
- every 10M until 100M: `20M, 30M, ..., 100M`;
- every 100M until 1B: `200M, 300M, ..., 1000M`.

The evaluation pipeline expects Hugging Face revisions named `chck_{count}M`. Early checkpoints are evaluated through zero-shot evaluation, while full fine-tuning is performed on the final checkpoint.

## 3. Official Multilingual Evaluation

Evaluation combines formal linguistic competence and downstream functionality. The official pipeline is available in [babylm-org/babylm-eval](https://github.com/babylm-org/babylm-eval), under the `multilingual` directory.

### Zero-Shot

- English: BLiMP, HellaSwag, MultiBLiMP, Winogrande, XStoryCloze, and GlobalPIQA.
- Dutch: BLiMP-NL, HellaSwag, MultiBLiMP, Winogrande, XCOMPS, XStoryCloze, and GlobalPIQA.
- Chinese: HellaSwag, Winogrande, XCOMPS, XStoryCloze, ZhoBLiMP, and GlobalPIQA.
- Hanzi Structure and Hanzi Pinyin are reported tasks but currently excluded from the primary averages.
- MECO measures how well word surprisal predicts human reading times; it is evaluated server-side and currently excluded from the primary average.

### Fine-Tuning

- English: ARC, Belebele, BMLama, MNLI, POS, SIB-200, TruthfulQA, and XNLI.
- Dutch: ARC, Belebele, BMLama, INCLUDE, MNLI, POS, SIB-200, and TruthfulQA.
- Chinese: ARC, Belebele, BMLama, INCLUDE, MNLI, POS, SIB-200, TruthfulQA, and XNLI.

POS uses a single multilingual fine-tuning setup with equal numbers of sentences per language. The remaining tasks are fine-tuned separately per language.

### Leaderboard Computation

The leaderboard implementation indicates that:

1. each language average uses a fixed denominator including all expected public tasks;
2. omitted tasks contribute zero;
3. the Multilingual score is the unweighted average of English, Dutch, and Chinese averages;
4. a completely omitted language also contributes zero.

The leaderboard screenshot `docs/leaderboard_babylm2026.png` includes:

- `Multilingual Average`;
- overall averages per language;
- zero-shot averages per language;
- fine-tuning averages per language.

The visible multilingual GPT-2 baseline achieves approximately `45.94` overall (`47.60` English, `46.92` Dutch, `43.31` Chinese). It is a useful reference point, but not a sufficient experimental control for a MoE-focused paper.

The leaderboard is dynamic and should be treated as a snapshot rather than a final ranking.

### Official Commands

From `babylm-eval/multilingual`:

```bash
bash scripts/eval_model_full.sh --model_name ORG/MODEL
bash scripts/eval_model_fast.sh --model_name ORG/MODEL
python scripts/collate_results.py --model_name ORG/MODEL --fast
```

Two artifacts are uploaded:

- public scores;
- raw predictions.

The model must be publicly accessible on Hugging Face.

## 4. Deliverables and Timeline

A competitive submission requires:

- a paper of up to eight pages, excluding references and appendices;
- a public Hugging Face-compatible model;
- intermediate checkpoints;
- official-format results and predictions;
- the multilingual corpus used, and a datasheet if a custom mixture is employed;
- reproducible hyperparameters and methodological decisions.

Public timeline:

- July 2026 direct submission deadline (verify exact date through official channels before submission);
- early August: reviews / ARR commitment;
- mid August: decisions;
- early September: camera-ready;
- October 24–29: EMNLP Workshop in Budapest.

## 5. Compliance Checklist for BabyMoE

- [ ] Do not load pretrained weights trained on data outside the allowed budget unless explicitly approved.
- [ ] Train the tokenizer only on permitted data or document official authorization.
- [ ] Freeze an exact training-document list before evaluation.
- [ ] Compute the language budget using Byte Premium and store the manifest.
- [ ] Track cumulative exposure, not only unique corpus size.
- [ ] Save `chck_1M` through `chck_1000M` at the required milestones.
- [ ] Export a causal `PreTrainedModel` capable of producing logits and scoring sequences.
- [ ] Run `eval_model_fast.sh` during development and `eval_model_full.sh` on the final model.
- [ ] Upload scores and predictions; avoid accidentally omitting tasks.
- [ ] Publish tokenizer, config, weights, code, datasheet, and seeds.
- [ ] Obtain written confirmation for any teacher-model or external-model usage in Multilingual.
- [ ] Verify the final submission deadline immediately before submission.
- [ ] Ensure AutoModel and AutoTokenizer compatibility.
- [ ] Record git commit, random seed, tokenizer hash, corpus manifest, Byte Premium accounting, and total exposure.
- [ ] Prevent evaluation-data leakage during pretraining, tokenizer training, filtering, and model selection.
- [ ] When rules are ambiguous, adopt the most conservative interpretation to minimize disqualification risk.