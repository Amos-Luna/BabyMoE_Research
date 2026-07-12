from __future__ import annotations

import numpy as np

from babymoe.analysis import ProbeInput, run_layerwise_probes


def test_routing_probe_detects_residual_language_information():
    groups = np.repeat(np.arange(40), 3)
    labels = np.tile(np.arange(3), 40)
    controls = np.full((len(labels), 5), "same", dtype=object)
    routing = np.zeros((len(labels), 2, 3), dtype=np.float32)
    routing[np.arange(len(labels)), 0, labels] = 1.0
    routing[:, 1, :] = 1 / 3
    result = run_layerwise_probes(
        ProbeInput(
            controls=controls,
            routing=routing,
            labels=labels,
            groups=groups,
            strata=np.full(len(labels), "same"),
        ),
        bootstrap_samples=20,
        permutation_samples=20,
        seed=42,
    )
    assert result["layers"][0]["delta_macro_auroc"] > 0.4
    assert abs(result["layers"][1]["delta_macro_auroc"]) < 0.05
