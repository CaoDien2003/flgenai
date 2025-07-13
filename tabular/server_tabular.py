# === IMPORT LIBRARIES ===
import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
import shap
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import platform
import psutil
import time
from datetime import datetime

# === SERVER CONFIGURATION ===
CONFIG = {
    "server_address": "0.0.0.0:9000",
    "test_csv_path": "dataset/new7030/test_server.csv",
    "target_column": "Exited",
    "max_feedback_per_client": 1000,
    "num_rounds": 30,
    "min_clients": 5,
    "server_log_dir": "./server_logs/exp01",
    "server_summary_filename": "summary_log.csv",
    "server_time_filename": "summary_time.csv",
    "server_config_filename": "server_config.json",
    "plateau_window": 3,
    "plateau_delta": 0.01,
    "min_feedback_round": 3,
    "shap_threshold": 0.05,
    "top_k_features": 10
}

CONFIG["server_feedback_dir"] = os.path.join(CONFIG["server_log_dir"], "feedback_logs")
os.makedirs(CONFIG["server_log_dir"], exist_ok=True)
os.makedirs(CONFIG["server_feedback_dir"], exist_ok=True)

server_log = []
recent_accuracies = []
feedback_storage = {}
time_log = []

print("[SERVER] Loading test dataset...")
test_df = pd.read_csv(CONFIG["test_csv_path"])
X_test = test_df.drop(columns=[CONFIG["target_column"]]).values
y_test = test_df[CONFIG["target_column"]].values
feature_names = test_df.drop(columns=[CONFIG["target_column"]]).columns.tolist()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
testloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)

def get_parameters(model):
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]

def set_parameters(model, parameters):
    model.load_state_dict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), parameters)
    })

def compute_shap_feedback(model, x_tensor, feature_names, threshold=0.05):
    model.eval()
    background = x_tensor[:100] if len(x_tensor) >= 100 else x_tensor
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_tensor, check_additivity=False)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    x_numpy = x_tensor.detach().cpu().numpy()

    feedback_list = []
    for shap_vec, real_vec in zip(shap_values, x_numpy):
        abs_shap = np.abs(shap_vec)
        passed_threshold_idx = [i for i, val in enumerate(abs_shap) if val >= threshold]
        sorted_idx = sorted(passed_threshold_idx, key=lambda i: abs_shap[i], reverse=True)
        top_k = CONFIG.get("top_k_features", 5)
        selected_idx = sorted_idx[:top_k]

        feat_dict = {
            name: float(real_val) if idx in selected_idx else np.nan
            for idx, (name, real_val) in enumerate(zip(feature_names, real_vec))
        }
        feedback_list.append(feat_dict)
    return feedback_list

def generate_feedback(client_id, model, dataloader, feature_names):
    model.eval()
    inputs_all, targets_all = [], []

    for inputs, targets in dataloader:
        outputs = model(inputs).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        for x, y_true, y_pred in zip(inputs, targets.int(), preds):
            if y_true.item() != y_pred.item():
                inputs_all.append(x)
                targets_all.append(int(y_true.item()))

    if not inputs_all:
        feedback_storage[client_id] = {"shap_feedback": [], "real_samples": []}
        return 0.0

    label_counts = Counter(targets_all)
    most_common_label, _ = label_counts.most_common(1)[0]

    filtered_samples = [
        (x, y) for x, y in zip(inputs_all, targets_all) if y == most_common_label
    ]

    if not filtered_samples:
        feedback_storage[client_id] = {"shap_feedback": [], "real_samples": []}
        return 0.0

    x_tensor = torch.stack([s[0] for s in filtered_samples])
    start_fb = time.time()
    shap_feedbacks = compute_shap_feedback(model, x_tensor, feature_names, CONFIG["shap_threshold"])
    fb_time = time.time() - start_fb

    shap_entries = []
    real_samples = []
    for shap_feat, (x_val, label) in zip(shap_feedbacks, filtered_samples):
        shap_entries.append({"features": shap_feat, "label": label})
        real_feature_dict = {name: float(val.item()) for name, val in zip(feature_names, x_val)}
        real_feature_dict[CONFIG["target_column"]] = label
        real_samples.append(real_feature_dict)

    feedback_storage[client_id] = {
        "shap_feedback": shap_entries[:CONFIG["max_feedback_per_client"]],
        "real_samples": real_samples[:CONFIG["max_feedback_per_client"]]
    }

    round_num = CONFIG.get("current_round", 0)
    feedback_path = os.path.join(CONFIG["server_feedback_dir"], f"feedback_round_{round_num}_client_{client_id}.json")
    with open(feedback_path, "w") as f:
        json.dump(feedback_storage[client_id], f, indent=2)
    return fb_time

def evaluate_fn(server_round, parameters, config, log_result=True):
    model = SimpleNN(X_test_tensor.shape[1])
    set_parameters(model, parameters)
    model.eval()
    correct, total, loss_total = 0, 0, 0.0
    criterion = nn.BCEWithLogitsLoss()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in testloader:
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
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    if log_result:
        recent_accuracies.append(acc)
        if len(recent_accuracies) > CONFIG["plateau_window"]:
            recent_accuracies.pop(0)
        server_log.append({
            "round": server_round,
            "accuracy": acc,
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        })

    return avg_loss, {"accuracy": acc}

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, testloader, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.testloader = testloader
        self.client_id_map = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.start_time = time.time()
        clients = list(client_manager.sample(CONFIG["min_clients"]))
        fit_ins_list = []
        for client_proxy in clients:
            client_id = self.client_id_map.get(client_proxy.cid, client_proxy.cid)
            feedback_bundle = feedback_storage.get(client_id, {})
            shap_feedback = feedback_bundle.get("shap_feedback", [])
            real_samples = feedback_bundle.get("real_samples", [])
            config = {
                "feedback": json.dumps(shap_feedback),
                "reference": json.dumps(real_samples),
                "server_round": server_round
            }
            fit_ins_list.append((client_proxy, fl.common.FitIns(parameters, config)))
        return fit_ins_list

    def aggregate_fit(self, server_round, results, failures):
        CONFIG["current_round"] = server_round
        start_agg = time.time()
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        agg_time = time.time() - start_agg
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

        acc = evaluate_fn(server_round, aggregated_weights, config={}, log_result=True)[1]["accuracy"]

        fb_time = 0.0
        plateau_triggered = False
        if server_round >= CONFIG["min_feedback_round"] and len(recent_accuracies) >= CONFIG["plateau_window"]:
            deltas = [abs(recent_accuracies[i+1] - recent_accuracies[i]) for i in range(-CONFIG["plateau_window"], -1)]
            if all(delta < CONFIG["plateau_delta"] for delta in deltas):
                plateau_triggered = True

        if plateau_triggered:
            for client_proxy, fit_res in results:
                client_id = fit_res.metrics.get("client_id", client_proxy.cid)
                self.client_id_map[client_proxy.cid] = client_id
                weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
                temp_model = SimpleNN(X_test_tensor.shape[1])
                set_parameters(temp_model, weights)
                fb_time += generate_feedback(client_id, temp_model, self.testloader, feature_names)

        set_parameters(self.model, aggregated_weights)
        round_time = time.time() - self.start_time
        time_log.append({
            "round": server_round,
            "round_time_sec": round(round_time, 2),
            "aggregate_time_sec": round(agg_time, 2),
            "feedback_time_sec": round(fb_time, 2)
        })
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

    def evaluate(self, server_round, parameters):
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        return evaluate_fn(server_round, ndarrays, config={}, log_result=True)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return None

def log_server_system_info():
    info = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "CPU Threads": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    print("[SERVER] System Info:")
    for k, v in info.items():
        print(f"[SERVER] {k}: {v}")

start_time = time.time()
log_server_system_info()
print("[SERVER] Server training started.")

def main():
    model = SimpleNN(X_test_tensor.shape[1])
    strategy = CustomFedAvg(
        model=model,
        testloader=testloader,
        min_fit_clients=CONFIG["min_clients"],
        min_available_clients=CONFIG["min_clients"],
        min_evaluate_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)),
        evaluate_fn=lambda r, p, c: evaluate_fn(r, p, c, log_result=True)
    )
    print("[SERVER] Starting Flower server...")
    fl.server.start_server(
        server_address=CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=CONFIG["num_rounds"]),
        strategy=strategy
    )
    log_path = os.path.join(CONFIG["server_log_dir"], CONFIG["server_summary_filename"])
    pd.DataFrame(server_log).to_csv(log_path, index=False)
    time_path = os.path.join(CONFIG["server_log_dir"], CONFIG["server_time_filename"])
    pd.DataFrame(time_log).to_csv(time_path, index=False)
    config_path = os.path.join(CONFIG["server_log_dir"], CONFIG["server_config_filename"])
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"[SERVER] Training summary saved to {log_path}")
    print(f"[SERVER] Time summary saved to {time_path}")
    print(f"[SERVER] Config saved to {config_path}")
    print(f"[SERVER] Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
