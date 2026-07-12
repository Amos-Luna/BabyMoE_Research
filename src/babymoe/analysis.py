"""Confound-controlled language probes and clustered statistical tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder


LANGUAGE_TO_ID = {"eng": 0, "nld": 1, "zho": 2}


def macro_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        roc_auc_score(
            labels,
            probabilities,
            labels=np.asarray([0, 1, 2]),
            multi_class="ovr",
            average="macro",
        )
    )


def _fit_probe(
    features: sparse.spmatrix,
    labels: np.ndarray,
    *,
    random_state: int,
) -> LogisticRegression:
    classifier = LogisticRegression(
        solver="saga",
        max_iter=500,
        tol=1e-3,
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
    )
    classifier.fit(features, labels)
    return classifier


def clustered_bootstrap_delta(
    *,
    labels: np.ndarray,
    control_probabilities: np.ndarray,
    augmented_probabilities: np.ndarray,
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values: list[float] = []
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(samples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[group] for group in sampled])
        if len(np.unique(labels[indices])) < 3:
            continue
        try:
            delta = macro_auroc(labels[indices], augmented_probabilities[indices])
            delta -= macro_auroc(labels[indices], control_probabilities[indices])
            values.append(delta)
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())  # type: ignore[return-value]


def within_stratum_permutation_pvalue(
    *,
    observed_delta: float,
    labels: np.ndarray,
    control_probabilities: np.ndarray,
    augmented_probabilities: np.ndarray,
    strata: np.ndarray,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    unique_strata = np.unique(strata)
    rows = {value: np.flatnonzero(strata == value) for value in unique_strata}
    exceedances = 0
    valid = 0
    for _ in range(samples):
        permuted = labels.copy()
        for indices in rows.values():
            if len(indices) > 1:
                permuted[indices] = rng.permutation(permuted[indices])
        if len(np.unique(permuted)) < 3:
            continue
        try:
            delta = macro_auroc(permuted, augmented_probabilities)
            delta -= macro_auroc(permuted, control_probabilities)
        except ValueError:
            continue
        valid += 1
        exceedances += int(delta >= observed_delta)
    return (exceedances + 1) / (valid + 1)


@dataclass
class ProbeInput:
    controls: np.ndarray
    routing: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    strata: np.ndarray


def run_layerwise_probes(
    data: ProbeInput,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_indices, test_indices = next(
        splitter.split(data.controls, data.labels, groups=data.groups)
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    control_train = encoder.fit_transform(data.controls[train_indices])
    control_test = encoder.transform(data.controls[test_indices])
    control_probe = _fit_probe(
        control_train, data.labels[train_indices], random_state=seed
    )
    control_probabilities = control_probe.predict_proba(control_test)
    control_score = macro_auroc(data.labels[test_indices], control_probabilities)

    layers = []
    for layer in range(data.routing.shape[1]):
        augmented_train = sparse.hstack(
            [control_train, sparse.csr_matrix(data.routing[train_indices, layer])],
            format="csr",
        )
        augmented_test = sparse.hstack(
            [control_test, sparse.csr_matrix(data.routing[test_indices, layer])],
            format="csr",
        )
        augmented_probe = _fit_probe(
            augmented_train,
            data.labels[train_indices],
            random_state=seed + layer + 1,
        )
        augmented_probabilities = augmented_probe.predict_proba(augmented_test)
        augmented_score = macro_auroc(
            data.labels[test_indices], augmented_probabilities
        )
        delta = augmented_score - control_score
        lower, upper = clustered_bootstrap_delta(
            labels=data.labels[test_indices],
            control_probabilities=control_probabilities,
            augmented_probabilities=augmented_probabilities,
            groups=data.groups[test_indices],
            samples=bootstrap_samples,
            seed=seed + 10_000 + layer,
        )
        pvalue = within_stratum_permutation_pvalue(
            observed_delta=delta,
            labels=data.labels[test_indices],
            control_probabilities=control_probabilities,
            augmented_probabilities=augmented_probabilities,
            strata=data.strata[test_indices],
            samples=permutation_samples,
            seed=seed + 20_000 + layer,
        )
        layers.append(
            {
                "layer": layer,
                "control_macro_auroc": control_score,
                "augmented_macro_auroc": augmented_score,
                "delta_macro_auroc": delta,
                "bootstrap_95_ci": [lower, upper],
                "within_stratum_permutation_pvalue": pvalue,
            }
        )
    return {
        "train_tokens": int(len(train_indices)),
        "test_tokens": int(len(test_indices)),
        "control_feature_columns": int(data.controls.shape[1]),
        "layers": layers,
    }
