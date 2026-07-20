from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .core import ConceptGraph, PriorityPolicy, Scenario, _build_features, generate_scenario, stable_softmax


FEATURE_NAMES = (
    "Demand pressure",
    "Backlog pressure",
    "Inventory risk",
    "Lead-time risk",
    "Capacity risk",
    "Reporting confidence",
)


def hospital_feature_tensor(features: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            features["demand_pressure"],
            features["backlog_pressure"],
            features["inventory_risk"],
            features["lead_risk"],
            features["capacity_risk"],
            features["reporting_confidence"],
        ],
        axis=-1,
    )


def normalized_hospital_graph(graph: ConceptGraph) -> np.ndarray:
    indices = graph.hospital_indices.ravel()
    a = np.abs(graph.adjacency[np.ix_(indices, indices)] * graph.edge_strength[np.ix_(indices, indices)])
    a = a + np.eye(a.shape[0])
    degree = np.maximum(a.sum(axis=1), 1e-12)
    d_inv = np.diag(1.0 / np.sqrt(degree))
    return d_inv @ a @ d_inv


@dataclass
class TrainingData:
    x: np.ndarray
    y: np.ndarray
    snapshot_ids: np.ndarray
    graph_features: np.ndarray


def make_training_data(
    graph: ConceptGraph,
    n_scenarios: int,
    days: int,
    master_seed: int,
    reporting_noise: float,
    missing_probability: float,
) -> TrainingData:
    rng = np.random.default_rng(master_seed + 119)
    ahat = normalized_hospital_graph(graph)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gids: list[np.ndarray] = []
    gx: list[np.ndarray] = []
    snapshot_id = 0
    for s in range(n_scenarios):
        scenario = generate_scenario(
            master_seed + 10000 + s,
            days=days,
            reporting_noise=reporting_noise,
            missing_probability=missing_probability,
        )
        for day in range(days):
            demand = scenario.observed_demand[day]
            depletion = np.clip(1.0 - day / (days * 1.4), 0.25, 1.0)
            hospital_inventory = scenario.initial_hospital_inventory * depletion * rng.uniform(0.55, 1.35, size=demand.shape)
            backlog = np.maximum(demand - hospital_inventory, 0.0) * rng.uniform(0.2, 1.3, size=demand.shape)
            dc_inventory = scenario.initial_dc_inventory * rng.uniform(0.35, 1.25, size=scenario.initial_dc_inventory.shape)
            inbound = rng.uniform(0.0, 0.7, size=demand.shape) * demand
            features = _build_features(scenario, day, hospital_inventory, dc_inventory, backlog, inbound)
            x = hospital_feature_tensor(features).reshape(-1, len(FEATURE_NAMES))
            criticality = (
                0.95 * x[:, 0]
                + 1.30 * x[:, 1]
                + 1.15 * x[:, 2]
                + 0.45 * x[:, 3]
                + 0.72 * x[:, 4]
                + 0.18 * (1.0 - x[:, 5])
            )
            y = 1.0 / (1.0 + np.exp(-3.0 * (criticality - 1.45)))
            graph_x = np.c_[x, ahat @ x, ahat @ ahat @ x]
            xs.append(x)
            gx.append(graph_x)
            ys.append(y)
            gids.append(np.full(x.shape[0], snapshot_id, dtype=int))
            snapshot_id += 1
    return TrainingData(np.vstack(xs), np.concatenate(ys), np.concatenate(gids), np.vstack(gx))


class SGCPolicy(PriorityPolicy):
    display_name = "SGC-GNN"

    def __init__(self, scaler: StandardScaler, model: MLPRegressor, temperature: float = 0.30):
        self.scaler = scaler
        self.model = model
        self.temperature = temperature

    def reset(self, scenario: Scenario, graph: ConceptGraph) -> None:
        super().reset(scenario, graph)
        self.ahat = normalized_hospital_graph(graph)

    def priority(self, features: dict[str, np.ndarray], day: int):
        h, r = features["demand_pressure"].shape
        x = hospital_feature_tensor(features).reshape(-1, len(FEATURE_NAMES))
        gx = np.c_[x, self.ahat @ x, self.ahat @ self.ahat @ x]
        pred = np.clip(self.model.predict(self.scaler.transform(gx)), 0.0, 1.0).reshape(h, r)
        return stable_softmax(pred, axis=0, temperature=self.temperature), {"score": pred, "iterations": 1, "convergence_gaps": np.array([0.0])}


class BayesianNetworkPolicy(PriorityPolicy):
    display_name = "Bayesian network"

    def __init__(self, prior: np.ndarray, likelihood: np.ndarray, bin_edges: np.ndarray, temperature: float = 0.30):
        self.prior = prior
        self.likelihood = likelihood
        self.bin_edges = bin_edges
        self.temperature = temperature

    def priority(self, features: dict[str, np.ndarray], day: int):
        h, r = features["demand_pressure"].shape
        x = hospital_feature_tensor(features).reshape(-1, len(FEATURE_NAMES))
        logp = np.tile(np.log(np.maximum(self.prior, 1e-12)), (x.shape[0], 1))
        for f in range(x.shape[1]):
            bins = np.clip(np.digitize(x[:, f], self.bin_edges[f]) - 1, 0, self.likelihood.shape[2] - 1)
            for cls in (0, 1):
                logp[:, cls] += np.log(np.maximum(self.likelihood[f, cls, bins], 1e-12))
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        posterior = (p[:, 1] / np.maximum(p.sum(axis=1), 1e-12)).reshape(h, r)
        return stable_softmax(posterior, axis=0, temperature=self.temperature), {"score": posterior, "iterations": 1, "convergence_gaps": np.array([0.0])}


class PPOPolicy(PriorityPolicy):
    display_name = "PPO"

    def __init__(self, scaler: StandardScaler, weights: np.ndarray, temperature: float = 0.30):
        self.scaler = scaler
        self.weights = weights
        self.temperature = temperature

    def priority(self, features: dict[str, np.ndarray], day: int):
        h, r = features["demand_pressure"].shape
        x = hospital_feature_tensor(features).reshape(-1, len(FEATURE_NAMES))
        z = np.c_[self.scaler.transform(x), np.ones(x.shape[0])]
        score = (1.0 / (1.0 + np.exp(-np.clip(z @ self.weights, -30.0, 30.0)))).reshape(h, r)
        return stable_softmax(score, axis=0, temperature=self.temperature), {"score": score, "iterations": 1, "convergence_gaps": np.array([0.0])}


@dataclass
class LearningBundle:
    sgc_policy: SGCPolicy
    bayesian_policy: BayesianNetworkPolicy
    ppo_policy: PPOPolicy
    training_history: pd.DataFrame
    calibration: pd.DataFrame


def _fit_bayesian_network(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = (y >= np.median(y)).astype(int)
    n_features = x.shape[1]
    n_bins = 3
    edges = np.zeros((n_features, n_bins + 1))
    likelihood = np.zeros((n_features, 2, n_bins))
    prior = np.array([(labels == 0).mean(), (labels == 1).mean()])
    for f in range(n_features):
        edges[f] = np.quantile(x[:, f], [0.0, 1 / 3, 2 / 3, 1.0])
        edges[f, 0] = -np.inf
        edges[f, -1] = np.inf
        bins = np.clip(np.digitize(x[:, f], edges[f]) - 1, 0, n_bins - 1)
        for cls in (0, 1):
            counts = np.bincount(bins[labels == cls], minlength=n_bins).astype(float) + 1.0
            likelihood[f, cls] = counts / counts.sum()
    return prior, likelihood, edges


def _fit_linear_ppo(
    x: np.ndarray,
    target: np.ndarray,
    interactions: int,
    master_seed: int,
) -> tuple[StandardScaler, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(master_seed + 771)
    scaler = StandardScaler().fit(x)
    z_all = np.c_[scaler.transform(x), np.ones(x.shape[0])]
    w = np.zeros(z_all.shape[1])
    value_w = np.zeros(z_all.shape[1])
    clip_eps = 0.20
    lr_actor = 0.025
    lr_value = 0.04
    batch = min(1024, max(128, interactions // 200))
    updates = max(1, int(np.ceil(interactions / batch)))
    rows: list[dict[str, float]] = []
    seen = 0
    for update in range(updates):
        idx = rng.integers(0, z_all.shape[0], size=batch)
        z = z_all[idx]
        t = target[idx]
        old_logits = np.clip(z @ w, -20.0, 20.0)
        old_prob = 1.0 / (1.0 + np.exp(-old_logits))
        actions = rng.binomial(1, old_prob)
        reward = actions * t + (1 - actions) * (1.0 - t) - 0.04 * np.abs(actions - old_prob)
        value = z @ value_w
        advantage = reward - value
        advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
        for _ in range(4):
            logits = np.clip(z @ w, -20.0, 20.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            logp = actions * np.log(prob + 1e-10) + (1 - actions) * np.log(1 - prob + 1e-10)
            old_logp = actions * np.log(old_prob + 1e-10) + (1 - actions) * np.log(1 - old_prob + 1e-10)
            ratio = np.exp(np.clip(logp - old_logp, -8.0, 8.0))
            active = np.where(advantage >= 0, ratio <= 1 + clip_eps, ratio >= 1 - clip_eps)
            coeff = advantage * ratio * active.astype(float)
            grad = z.T @ (coeff * (actions - prob)) / batch
            entropy_grad = z.T @ ((0.5 - prob) * 0.002) / batch
            w += lr_actor * (grad + entropy_grad)
        value_error = (z @ value_w) - reward
        value_w -= lr_value * (z.T @ value_error) / batch
        seen += batch
        if update % max(1, updates // 100) == 0 or update == updates - 1:
            pred = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -20.0, 20.0)))
            rows.append(
                {
                    "interaction": min(seen, interactions),
                    "mean_reward": float(reward.mean()),
                    "policy_target_correlation": float(np.corrcoef(pred, t)[0, 1]),
                    "policy_entropy": float(np.mean(-pred * np.log(pred + 1e-10) - (1 - pred) * np.log(1 - pred + 1e-10))),
                }
            )
    return scaler, w, pd.DataFrame(rows)


def train_learning_baselines(
    graph: ConceptGraph,
    n_scenarios: int,
    days: int,
    interactions: int,
    master_seed: int,
    reporting_noise: float,
    missing_probability: float,
    temperature: float,
) -> LearningBundle:
    data = make_training_data(graph, n_scenarios, days, master_seed, reporting_noise, missing_probability)

    sgc_scaler = StandardScaler().fit(data.graph_features)
    sgc = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=220,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=master_seed,
    )
    sgc.fit(sgc_scaler.transform(data.graph_features), data.y)
    sgc_pred = np.clip(sgc.predict(sgc_scaler.transform(data.graph_features)), 0.0, 1.0)

    prior, likelihood, edges = _fit_bayesian_network(data.x, data.y)
    ppo_scaler, ppo_weights, ppo_history = _fit_linear_ppo(data.x, data.y, interactions, master_seed)

    calibration = pd.DataFrame(
        [
            {"model": "SGC-GNN", "training_samples": len(data.y), "target_correlation": np.corrcoef(sgc_pred, data.y)[0, 1], "training_iterations": sgc.n_iter_},
            {"model": "Bayesian network", "training_samples": len(data.y), "target_correlation": np.nan, "training_iterations": 1},
            {"model": "PPO", "training_samples": interactions, "target_correlation": ppo_history.iloc[-1]["policy_target_correlation"], "training_iterations": len(ppo_history)},
        ]
    )
    return LearningBundle(
        sgc_policy=SGCPolicy(sgc_scaler, sgc, temperature),
        bayesian_policy=BayesianNetworkPolicy(prior, likelihood, edges, temperature),
        ppo_policy=PPOPolicy(ppo_scaler, ppo_weights, temperature),
        training_history=ppo_history,
        calibration=calibration,
    )

