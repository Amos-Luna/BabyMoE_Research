# Group #5
## Topic: BabyMoE: Language-Specific Expert Routing in Small Multilingual Language Models 

Multilingual Track

- Research Question: When training a small MoE language model under the BabyLM multilingual data budget, do experts specialize by language, or do they learn shared cross-lingual computation? More specifically, does expert routing reveal language-specific versus language-agnostic representations across English, Dutch, and Chinese?

- MoEP: Modular Expert Paths for Sample-Efficient Language Modeling (https://aclanthology.org/2025.babylm-main.39.pdf) shows that sparse expert routing is feasible in BabyLM-scale training and can improve sample efficiency over dense GPT-2-style baselines. Separately, recent work (https://arxiv.org/pdf/2601.14050) on multilingual MoE LLMs shows that expert routing in large models is structured by language, with some experts serving language-specific roles and middle layers acting as shared multilingual capacity hubs. We want to test whether this phenomenon appears in small multilingual models trained from scratch, or whether it requires large-scale pretraining to emerge.


## BabyMoE Research Proposal

### Summary

**BabyMoE** asks whether a small sparse language model trained from scratch develops language-specific experts or shared cross-lingual computation. We will train on the permitted English, Dutch, and Chinese BabyBabelLM data and analyze routing throughout learning. The paper's contribution is not simply an expert-by-language heatmap: it is a controlled test of whether apparent language specialization remains after accounting for token identity, script, frequency, punctuation, position, and domain. This complements BabyLM-scale evidence that sparse routing can be sample-efficient ([MoEP](https://aclanthology.org/2025.babylm-main.39/)) and tests whether patterns reported in large pretrained multilingual MoEs ([Multilingual Routing in Mixture-of-Experts, 2025](https://arxiv.org/abs/2510.04694); [Tsai et al., 2026](https://arxiv.org/abs/2601.14050)) emerge under a human-scale data budget.

For competition compliance, all submitted models and the joint tokenizer will be trained from scratch using only permitted data, within the 100M Byte-Premium-adjusted corpus budget and 1B cumulative-exposure cap. Required intermediate checkpoints will be retained.

### Impactful Hypothesis

Under limited multilingual input, learned sparse routing develops a layerwise division of labor: early and late layers remain more language-sensitive, while middle layers share more computation across languages. If this structure predicts better cross-lingual performance, it would suggest that multilingual modularity is an acquisition phenomenon that does not require billion-parameter models or massive pretraining. If it does not appear, the result would identify scale or post-training as a necessary condition for the patterns reported in large MoEs.

### Proxy, Testable Hypothesis

On balanced held-out data, routing features will predict language significantly better than token-level controls alone, showing residual language organization rather than only script or vocabulary clustering. On parallel or meaning-matched examples, English–Dutch routing distributions should be more similar than English–Chinese and Dutch–Chinese distributions, with the smallest cross-language divergence in middle layers. These effects should replicate across seeds and emerge consistently across exposure checkpoints. A secondary prediction is that checkpoints with greater middle-layer sharing obtain better per-language scores in the official BabyLM evaluation, after controlling for exposure and validation loss.

### Minimum Viable Product

1. **Data and tokenizer:** use a fixed, auditable BabyBabelLM mixture and a joint tokenizer trained only on that mixture; keep analysis and evaluation examples out of pretraining.
2. **Models:** train a small decoder-only MoE from scratch (for example, four FFN experts with top-2 routing) and an active-parameter/FLOP-matched dense baseline using the same tokenizer, data order, optimizer, and exposure budget. Use three seeds for the short pilot if feasible.
3. **Developmental measurements:** save the required checkpoints and log per-layer router probabilities, selected experts, entropy, expert load, and dropped-token rate.
4. **Controlled analysis:** compare a classifier using token/script/frequency/position/domain features against the same classifier plus routing features. Measure expert-distribution divergence on balanced natural and analysis-only parallel data. Include a shuffled-routing null and a fixed random-router control if compute permits.
5. **Statistics and performance:** use document-level bootstrap confidence intervals and within-stratum permutation tests. Compare validation loss, learning speed, and official English, Dutch, and Chinese scores against the dense baseline.
6. **Decision rule:** proceed to the full run if residual routing information and the layerwise pattern replicate in at least two of three pilot seeds. A controlled null result remains useful if raw language heatmaps disappear after token and script controls.

### Possible Extensions

- Regularize only middle-layer routers toward cross-lingual sharing and test whether this causally improves transfer.
- Compare balanced and skewed language mixtures while preserving Byte Premium accounting.
- Add an always-active shared expert or vary the number of experts by layer.
- Intervene on routing at inference time to test whether shared versus language-specific experts affect performance.
- Compare MoE expert specialization with language-sensitive neurons in the dense baseline.

### Route to Value

Scientifically, the project tests whether multilingual expert organization is learned from limited input or is an artifact of scale, tokenization, or post-training. Practically, it can show BabyLM participants when sparse capacity helps, when it merely partitions scripts, and whether middle-layer sharing is a useful design target. The outputs—a compliant model, developmental checkpoints, routing logs, and controlled analyses—support both a competitive Multilingual submission and a standalone mechanistic paper.

### Possible Failure Modes

- **Tokenizer/script confounding:** Chinese tokens may appear language-specific even when the router only recognizes character identity.
- **Domain or mixture confounding:** different source distributions may be mistaken for linguistic specialization.
- **Router collapse or forced balance:** a few experts may dominate, or the balancing loss may hide genuine preferences.
- **Capacity artifacts:** token dropping, sequence position, or unequal expert utilization may create false patterns.
- **No specialization at this scale:** the model may be too small or undertrained; this is informative only with adequate controls and power.
- **Specialization without utility:** clear language routing may not improve loss, sample efficiency, or official scores.
- **Unfair baseline:** parameter count, active FLOPs, data order, or optimization differences may explain any MoE advantage.
- **Overclaiming:** correlational routing patterns do not by themselves establish linguistic representations or causal transfer.

### Why Has This Gap Remained, and Will It Remain Open?

A search of the literature as of July 12, 2026 found close precedents, not proof that nobody has studied the question. BabyLM MoE work has mainly tested sample efficiency in the English-focused tracks, while multilingual routing studies have mainly analyzed large pretrained or post-trained models. The exact intersection—trilingual training from scratch under the new BabyLM 2026 budget, with developmental checkpoints and token-level confound controls—therefore appears underexplored.

There is no strong reason to assume nobody else will do this soon: the Multilingual track is new, the relevant large-model results are recent, and the experiment is accessible. The novelty window is consequently narrow. A defensible paper must contribute checkpoint-level emergence, rigorous controls, and a link to functional performance or a causal intervention; reproducing an expert-language heatmap alone is unlikely to be novel.

## Official and Primary Sources

1. BabyLM 2026 Guidelines.
2. BabyLM 2026 Homepage and Schedule.
3. BabyLM 2026 Timeline.
4. Choshen et al. (2026), *BabyLM Turns 4 and Goes Multilingual*.
5. Official BabyLM 2026 Evaluation Repository.
6. BabyLM 2026 Leaderboard.
7. Jumelet et al. (2026), *BabyBabelLM*.
8. Arnett et al. (2024), *A Bit of a Problem: Measurement Disparities in Dataset Sizes across Languages*.