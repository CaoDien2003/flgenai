"""Federated learning server for the tabular churn task (Flower + SHAP).

The server runs FedAvg over the clients, evaluates the aggregated model on a
held-out test set every round, and writes per-round metrics and timings to CSV.

Once the global accuracy plateaus it builds SHAP feedback for every client: the
test samples that client's model misclassified, with all features below the SHAP
importance threshold masked out (NaN). Clients receive that masked feedback with
the next fit instruction and use it to steer their synthetic data generation.

    python server_tabular.py
"""

import json
import os
import platform
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import psutil
import shap
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# === SERVER CONFIGURATION ===
CONFIG = {
    "server_address": "0.0.0.0:9000",
    "test_csv_path": "dataset/7030/test_server.csv",
    "target_column": "Exited",
    "max_feedback_per_client": 1000,
    "num_rounds": 30,
    "min_clients": 1,
    "log_dir": "logs/exp01",
    "server_summary_filename": "summary_log.csv",
    "server_time_filename": "summary_time.csv",
    "server_config_filename": "server_config.json",
    "plateau_window": 3,
    "plateau_delta": 0.01,
    "min_feedback_round": 3,
    "shap_threshold": 0.05,
    "top_k_features": 10
}

# Every artefact of one run lives under log_dir: the server writes to <log_dir>/server/,
# each client to <log_dir>/clients/<client-id>/ (pass the same path as --log-dir).
CONFIG["server_log_dir"] = os.path.join(CONFIG["log_dir"], "server")
CONFIG["server_feedback_dir"] = os.path.join(CONFIG["server_log_dir"], "feedback_logs")


# === DATA & MODEL ===
def load_test_data() -> Tuple[DataLoader, int, List[str]]:
    """Load the global test set; returns the loader, the input size and feature names."""
    test_df = pd.read_csv(CONFIG["test_csv_path"])
    features = test_df.drop(columns=[CONFIG["target_column"]])
    feature_names = features.columns.tolist()

    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(test_df[CONFIG["target_column"]].values, dtype=torch.float32)
    testloader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=32)
    return testloader, x_tensor.shape[1], feature_names


class SimpleNN(nn.Module):
    """Global model; the clients train an identical architecture."""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    model.load_state_dict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), parameters)
    })


# === SHAP FEEDBACK ===
def compute_shap_feedback(model: nn.Module, x_tensor: torch.Tensor,
                          feature_names: List[str], threshold: float) -> List[Dict[str, float]]:
    """Mask every feature whose |SHAP value| is below the threshold.

    Only the top-k most important features that pass the threshold keep their real
    value; the rest are sent as NaN so no raw sample leaves the server intact.
    """
    model.eval()
    background = x_tensor[:100] if len(x_tensor) >= 100 else x_tensor
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_tensor, check_additivity=False)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    x_numpy = x_tensor.detach().cpu().numpy()
    top_k = CONFIG.get("top_k_features", 5)

    feedback_list = []
    for shap_vec, real_vec in zip(shap_values, x_numpy):
        abs_shap = np.abs(shap_vec)
        passed_threshold_idx = [i for i, val in enumerate(abs_shap) if val >= threshold]
        sorted_idx = sorted(passed_threshold_idx, key=lambda i: abs_shap[i], reverse=True)
        selected_idx = sorted_idx[:top_k]

        feedback_list.append({
            name: float(real_val) if idx in selected_idx else np.nan
            for idx, (name, real_val) in enumerate(zip(feature_names, real_vec))
        })
    return feedback_list


# === STRATEGY ===
class FeedbackFedAvg(fl.server.strategy.FedAvg):
    """FedAvg plus centralized evaluation and plateau-triggered SHAP feedback."""

    def __init__(self, model: nn.Module, testloader: DataLoader,
                 feature_names: List[str], input_size: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model
        self.testloader = testloader
        self.feature_names = feature_names
        self.input_size = input_size

        self.client_id_map: Dict[str, str] = {}
        self.feedback_storage: Dict[str, Dict[str, list]] = {}
        self.metrics_log: List[Dict[str, Any]] = []
        self.time_log: List[Dict[str, Any]] = []
        self.recent_accuracies: List[float] = []
        self.current_round = 0
        self.round_start = time.time()

    # --- Evaluation ---
    def evaluate_parameters(self, server_round: int, parameters: List[np.ndarray],
                            log_result: bool = True) -> Tuple[float, Dict[str, float]]:
        """Evaluate the given weights on the test set, optionally recording the metrics."""
        model = SimpleNN(self.input_size)
        set_parameters(model, parameters)
        model.eval()

        criterion = nn.BCEWithLogitsLoss()
        correct, total, loss_total = 0, 0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in self.testloader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                correct += (predicted == targets.int()).sum().item()
                total += len(targets)
                loss_total += loss.item() * len(targets)
                all_preds.extend(predicted.tolist())
                all_targets.extend(targets.int().tolist())

        acc = correct / total
        avg_loss = loss_total / total

        if log_result:
            tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
            self.recent_accuracies.append(acc)
            if len(self.recent_accuracies) > CONFIG["plateau_window"]:
                self.recent_accuracies.pop(0)
            self.metrics_log.append({
                "round": server_round,
                "accuracy": acc,
                "loss": avg_loss,
                "precision": precision_score(all_targets, all_preds, zero_division=0),
                "recall": recall_score(all_targets, all_preds, zero_division=0),
                "f1_score": f1_score(all_targets, all_preds, zero_division=0),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            })

        return avg_loss, {"accuracy": acc}

    def is_plateau(self, server_round: int) -> bool:
        """True once the recent accuracies stop moving by more than plateau_delta."""
        if server_round < CONFIG["min_feedback_round"]:
            return False
        if len(self.recent_accuracies) < CONFIG["plateau_window"]:
            return False
        window = CONFIG["plateau_window"]
        deltas = [abs(self.recent_accuracies[i + 1] - self.recent_accuracies[i])
                  for i in range(-window, -1)]
        return all(delta < CONFIG["plateau_delta"] for delta in deltas)

    # --- Feedback ---
    def generate_feedback(self, client_id: str, model: nn.Module) -> float:
        """Build and store SHAP feedback from this client's misclassified samples.

        Returns the time spent inside SHAP, so it can be reported separately.
        """
        model.eval()
        inputs_all, targets_all = [], []

        for inputs, targets in self.testloader:
            outputs = model(inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            for x, y_true, y_pred in zip(inputs, targets.int(), preds):
                if y_true.item() != y_pred.item():
                    inputs_all.append(x)
                    targets_all.append(int(y_true.item()))

        if not inputs_all:
            self.feedback_storage[client_id] = {"shap_feedback": [], "real_samples": []}
            return 0.0

        # Keep only the majority error class, so the feedback pulls in one direction
        most_common_label, _ = Counter(targets_all).most_common(1)[0]
        filtered_samples = [(x, y) for x, y in zip(inputs_all, targets_all) if y == most_common_label]
        if not filtered_samples:
            self.feedback_storage[client_id] = {"shap_feedback": [], "real_samples": []}
            return 0.0

        x_tensor = torch.stack([sample[0] for sample in filtered_samples])
        start_fb = time.time()
        shap_feedbacks = compute_shap_feedback(model, x_tensor, self.feature_names,
                                               CONFIG["shap_threshold"])
        fb_time = time.time() - start_fb

        shap_entries, real_samples = [], []
        for shap_feat, (x_val, label) in zip(shap_feedbacks, filtered_samples):
            shap_entries.append({"features": shap_feat, "label": label})
            real_feature_dict = {name: float(val.item()) for name, val in zip(self.feature_names, x_val)}
            real_feature_dict[CONFIG["target_column"]] = label
            real_samples.append(real_feature_dict)

        limit = CONFIG["max_feedback_per_client"]
        self.feedback_storage[client_id] = {
            "shap_feedback": shap_entries[:limit],
            "real_samples": real_samples[:limit]
        }

        feedback_path = os.path.join(CONFIG["server_feedback_dir"],
                                     f"feedback_round_{self.current_round}_client_{client_id}.json")
        with open(feedback_path, "w") as f:
            json.dump(self.feedback_storage[client_id], f, indent=2)
        return fb_time

    # --- Flower hooks ---
    def configure_fit(self, server_round: int, parameters: fl.common.Parameters,
                      client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[Any, fl.common.FitIns]]:
        self.round_start = time.time()
        fit_ins_list = []
        for client_proxy in client_manager.sample(CONFIG["min_clients"]):
            client_id = self.client_id_map.get(client_proxy.cid, client_proxy.cid)
            feedback_bundle = self.feedback_storage.get(client_id, {})
            config = {
                "feedback": json.dumps(feedback_bundle.get("shap_feedback", [])),
                "reference": json.dumps(feedback_bundle.get("real_samples", [])),
                "server_round": server_round
            }
            fit_ins_list.append((client_proxy, fl.common.FitIns(parameters, config)))
        return fit_ins_list

    def aggregate_fit(self, server_round: int, results: List[Any],
                      failures: List[Any]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        self.current_round = server_round

        start_agg = time.time()
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        agg_time = time.time() - start_agg
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

        # This is the single recording point for a training round: the plateau check below
        # needs this round's accuracy, and evaluate() deliberately does not record again.
        self.evaluate_parameters(server_round, aggregated_weights, log_result=True)

        fb_time = 0.0
        if self.is_plateau(server_round):
            for client_proxy, fit_res in results:
                client_id = fit_res.metrics.get("client_id", client_proxy.cid)
                self.client_id_map[client_proxy.cid] = client_id
                client_model = SimpleNN(self.input_size)
                set_parameters(client_model, fl.common.parameters_to_ndarrays(fit_res.parameters))
                fb_time += self.generate_feedback(client_id, client_model)

        set_parameters(self.model, aggregated_weights)
        self.time_log.append({
            "round": server_round,
            "round_time_sec": round(time.time() - self.round_start, 2),
            "aggregate_time_sec": round(agg_time, 2),
            "feedback_time_sec": round(fb_time, 2)
        })
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

    def evaluate(self, server_round: int,
                 parameters: fl.common.Parameters) -> Tuple[float, Dict[str, float]]:
        """Centralized evaluation, called by Flower before round 1 and after every round.

        Only round 0 (the initial parameters) is recorded here; aggregate_fit already
        recorded the same weights for rounds >= 1, so logging both would duplicate every
        round in summary_log.csv and halve the effective plateau window.
        """
        return self.evaluate_parameters(server_round, fl.common.parameters_to_ndarrays(parameters),
                                        log_result=(server_round == 0))

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters,
                           client_manager: fl.server.client_manager.ClientManager) -> None:
        return None  # Evaluation is centralized; clients are never asked to evaluate

    # --- Reporting ---
    def save_reports(self, total_time: float) -> None:
        log_dir = CONFIG["server_log_dir"]
        log_path = os.path.join(log_dir, CONFIG["server_summary_filename"])
        time_path = os.path.join(log_dir, CONFIG["server_time_filename"])
        config_path = os.path.join(log_dir, CONFIG["server_config_filename"])

        pd.DataFrame(self.metrics_log).to_csv(log_path, index=False)
        pd.DataFrame(self.time_log).to_csv(time_path, index=False)

        CONFIG["total_time_sec"] = round(total_time, 2)
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=2)

        print(f"[SERVER] Training summary saved to {log_path}")
        print(f"[SERVER] Time summary saved to {time_path}")
        print(f"[SERVER] Config saved to {config_path}")
        print(f"[SERVER] Total execution time: {total_time:.2f} seconds")


def log_system_info() -> None:
    info = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "CPU Threads": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }
    print("[SERVER] System Info:")
    for key, value in info.items():
        print(f"[SERVER] {key}: {value}")


# === MAIN ===
def main() -> None:
    start_time = time.time()
    os.makedirs(CONFIG["server_log_dir"], exist_ok=True)
    os.makedirs(CONFIG["server_feedback_dir"], exist_ok=True)

    print(f"[SERVER] Logging to {CONFIG['server_log_dir']}")
    print("[SERVER] Loading test dataset...")
    testloader, input_size, feature_names = load_test_data()

    log_system_info()
    print("[SERVER] Server training started.")

    model = SimpleNN(input_size)
    strategy = FeedbackFedAvg(
        model=model,
        testloader=testloader,
        feature_names=feature_names,
        input_size=input_size,
        min_fit_clients=CONFIG["min_clients"],
        min_available_clients=CONFIG["min_clients"],
        min_evaluate_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model))
    )

    print("[SERVER] Starting Flower server...")
    fl.server.start_server(
        server_address=CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=CONFIG["num_rounds"]),
        strategy=strategy
    )

    strategy.save_reports(time.time() - start_time)


if __name__ == "__main__":
    main()
