"""Federated learning client for the tabular churn task (Flower + PyTorch + SDV).

Each client trains a small MLP on its private CSV shard. When the server detects
an accuracy plateau it returns SHAP feedback: misclassified test samples in which
every feature below the importance threshold has been masked out (NaN). The
client then

1. restores the masked features with a TVAE fitted on its own data,
2. samples the pre-trained TVAE generator and keeps only the rows that are
   cosine-similar to the restored feedback,
3. trains on its own data plus the restored and selected synthetic rows.

Run one process per client, each with its own id and data shard:

    python client_tabular.py --client-id client1 --train-csv dataset/7030/train1.csv \
        --server-address 127.0.0.1:9000
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
# Values that identify this client (id, data shard, server address) are CLI flags;
# everything below is experiment tuning and is edited here.
CONFIG = {
    "client_id": "client1",  # Default id, overridden by --client-id
    "train_csv_path": "dataset/7030/train1.csv",  # Default shard, overridden by --train-csv
    "server_address": "127.0.0.1:9000",  # Default address, overridden by --server-address
    "target_column": "Exited",  # Label column in the training CSV
    "log_dir": "logs/exp01",  # Run log folder, overridden by --log-dir;
                              # this client writes to <log_dir>/clients/<client-id>/
    "gen_model_path": "tvae_gen.pkl",  # Path to the pre-trained TVAE generator
    "epoch": 50,  # Number of training epochs
    "batch_size": 32,  # Mini-batch size for training
    "learning_rate": 0.001,  # Adam learning rate
    "gen_multiplier": 3.0,  # Synthetic data size = multiplier x original size
    "cosine_sim_threshold": 0.9,  # Cosine similarity threshold for filtering
    "max_gen_attempts": 10,  # Max generate-and-filter rounds before giving up
    "use_feedback": True,  # True = generate based on server feedback; False = ignore feedback
    "gen_multiplier_fixed": 0.1,  # Synthetic data ratio ( use_feedback = False)
                                  # Final Synthetic data ( use_feedback = True)
}


# === CLI & LOGGING ===
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client-id", default=CONFIG["client_id"],
                        help="Unique client id; the server keys its feedback on it")
    parser.add_argument("--train-csv", default=CONFIG["train_csv_path"],
                        help="Path to this client's local training CSV")
    parser.add_argument("--server-address", default=CONFIG["server_address"],
                        help="Address of the Flower server, e.g. 127.0.0.1:9000")
    parser.add_argument("--log-dir", default=CONFIG["log_dir"],
                        help="Run log folder; use the same one as the server, e.g. logs/exp01")
    return parser.parse_args()


def setup_logging(log_dir: str) -> str:
    """Log to a timestamped file inside log_dir and return its path."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # SDV reports every fit/sample call on this logger; keep it out of the client log.
    logging.getLogger("SingleTableSynthesizer").propagate = False
    return log_path


# === MODEL DEFINITION ===
class SimpleNN(nn.Module):
    """Same architecture as the server's global model; weights are exchanged as-is."""

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


# === TVAE SYNTHESIZERS ===
def fit_restore_synthesizer(df: pd.DataFrame) -> TVAESynthesizer:
    """Fit a TVAE on the local data; used to fill the features masked by SHAP."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(df)
    return synthesizer


def load_generator(path: str) -> TVAESynthesizer:
    """Unpickle the pre-trained TVAE generator.

    tvae_gen.pkl was saved from an environment with numpy >= 2.0, which pickles
    RandomState / BitGenerator objects in a format numpy 1.x cannot rebuild
    (torch 2.2 pins numpy < 2). Temporarily patch numpy's reconstructors so the
    legacy random state is accepted, then restore them.
    """
    import numpy.random as npr
    import numpy.random._pickle as np_pickle
    from numpy.random import BitGenerator, Generator, RandomState

    def _setstate(self, state):
        self.state = state[0] if isinstance(state, tuple) else state

    def _bit_generator(bit_generator=None, **kwargs):
        if isinstance(bit_generator, BitGenerator):
            return bit_generator
        base = bit_generator if isinstance(bit_generator, type) else getattr(
            npr, bit_generator if isinstance(bit_generator, str) else "MT19937")
        # Keep the original class name: numpy validates state["bit_generator"] against it
        return type(base.__name__, (base,), {"__setstate__": _setstate})()

    saved = (np_pickle.__bit_generator_ctor,
             np_pickle.__randomstate_ctor,
             np_pickle.__generator_ctor)
    np_pickle.__bit_generator_ctor = _bit_generator
    np_pickle.__randomstate_ctor = lambda bg=None, **kw: RandomState(_bit_generator(bg))
    np_pickle.__generator_ctor = lambda bg=None, **kw: Generator(_bit_generator(bg))
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        (np_pickle.__bit_generator_ctor,
         np_pickle.__randomstate_ctor,
         np_pickle.__generator_ctor) = saved


# === FEEDBACK HANDLING ===
def feedback_to_dataframe(shap_feedback: List[Dict[str, Any]], target_column: str) -> pd.DataFrame:
    """Turn the server's [{features, label}, ...] payload into a DataFrame."""
    return pd.DataFrame([{**item["features"], target_column: item["label"]} for item in shap_feedback])


def restore_missing_values(fb_df: pd.DataFrame, synthesizer: TVAESynthesizer) -> pd.DataFrame:
    """Fill the features the server masked out with TVAE-sampled values."""
    restored_df = fb_df.copy()
    for col in restored_df.columns:
        if restored_df[col].isna().sum() > 0:
            fill_values = synthesizer.sample(len(restored_df))[col]
            restored_df[col] = restored_df[col].fillna(fill_values)
    return restored_df


def match_by_masked_features(gen_df: pd.DataFrame, fb_df: pd.DataFrame,
                             cosine_threshold: float) -> pd.DataFrame:
    """Keep the generated rows that are cosine-similar to any feedback row.

    Only the features present (non-NaN) in a feedback row take part in the
    comparison, so partially masked feedback can still be matched.
    """
    matched_rows = []
    for _, row in fb_df.iterrows():
        mask = ~row.isna()  # Mask of present (non-missing) features
        if mask.sum() == 0:
            continue
        fb_vec = row[mask].values.reshape(1, -1)
        gen_filtered = gen_df[mask.index[mask]]
        sims = cosine_similarity(gen_filtered.values, fb_vec)[:, 0]
        selected = gen_df[sims >= cosine_threshold]
        matched_rows.extend(selected.to_dict(orient="records"))
    return pd.DataFrame(matched_rows)


def collect_synthetic_samples(generator: TVAESynthesizer, restored_df: pd.DataFrame,
                              base_size: int) -> pd.DataFrame:
    """Generate and filter repeatedly until enough samples match the feedback."""
    required = int(CONFIG["gen_multiplier_fixed"] * base_size)
    batch_size = int(CONFIG["gen_multiplier"] * base_size)
    total_selected = pd.DataFrame()

    for attempt in range(CONFIG["max_gen_attempts"]):  # Avoid infinite loop
        gen_batch = generator.sample(batch_size)
        filtered = match_by_masked_features(gen_batch, restored_df, CONFIG["cosine_sim_threshold"])
        total_selected = pd.concat([total_selected, filtered]).drop_duplicates().reset_index(drop=True)

        logging.info(f"[GEN-{attempt + 1}] Filtered: {len(filtered)} | Total Selected: {len(total_selected)}")
        if len(total_selected) >= required:
            break

    selected_samples = total_selected.drop_duplicates().head(required)  # Truncate to exact required number
    logging.info(f"[FINAL] Selected {len(selected_samples)} synthetic samples (required {required})")
    return selected_samples


# === LOCAL TRAINING ===
def train_local_model(model: nn.Module, train_df: pd.DataFrame) -> int:
    """Train the model on train_df in place and return the number of samples used."""
    features = train_df.drop(columns=[CONFIG["target_column"]])
    labels = train_df[CONFIG["target_column"]]

    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                              batch_size=CONFIG["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    epochs = CONFIG["epoch"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logging.info(f"Epoch {epoch + 1} / {epochs} - Loss: {avg_loss:.4f}")
            print(f"[CLIENT] Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

    return len(x_tensor)


# === FLOWER CLIENT DEFINITION ===
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model: nn.Module, train_df: pd.DataFrame,
                 restore_synthesizer: TVAESynthesizer, generator: TVAESynthesizer):
        self.client_id = client_id
        self.model = model
        self.train_df = train_df
        self.restore_synthesizer = restore_synthesizer
        self.generator = generator
        # Rebuilt from train_df at every fit() call, so restored/synthetic rows never
        # accumulate across rounds; it is kept as state only to report the last size.
        self.train_data = train_df.copy()

    def get_parameters(self, config: Dict[str, Any]) -> List[Any]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[Any]) -> None:
        state_dict = self.model.state_dict()
        for key, value in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(value)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: List[Any], config: Dict[str, Any]) -> Tuple[List[Any], int, Dict[str, Any]]:
        self.set_parameters(parameters)

        shap_feedback = json.loads(config.get("feedback", "[]"))
        restored_df = pd.DataFrame()
        selected_samples = pd.DataFrame()

        if CONFIG["use_feedback"] and shap_feedback:
            logging.info(f"Using feedback: {len(shap_feedback)} samples")
            fb_df = feedback_to_dataframe(shap_feedback, CONFIG["target_column"])

            # === RESTORE MISSING VALUES USING TVAE
            restored_df = restore_missing_values(fb_df, self.restore_synthesizer)
            logging.info(f"NaN before restore: {fb_df.isna().sum().sum()} "
                         f"| after restore: {restored_df.isna().sum().sum()}")

            # === REPEATEDLY GENERATE & FILTER SYNTHETIC SAMPLES UNTIL ENOUGH SELECTED
            selected_samples = collect_synthetic_samples(self.generator, restored_df, len(self.train_df))

            self.train_data = pd.concat([self.train_df, restored_df, selected_samples], ignore_index=True)
            print(f'[DEBUG] DATA TRAIN CLIENT {len(self.train_df)}')
            print(f'[DEBUG] SYNTHETIC SAMPLES {len(selected_samples)}')
            print(f'[DEBUG] RESTORED SAMPLES {len(restored_df)}')
        else:
            logging.info("Gen ignore feedback")

            # === GENERATE FIXED SYNTHETIC DATA (no feedback)
            num_gen = int(CONFIG["gen_multiplier_fixed"] * len(self.train_df))
            gen_fixed = self.generator.sample(num_gen)
            logging.info(f"Generated {num_gen} synthetic samples ignore feedback")

            # === MERGE ALL DATA
            self.train_data = pd.concat([self.train_df, gen_fixed], ignore_index=True)

        logging.info(f"Final train size: Original={len(self.train_df)} | Restored={len(restored_df)} "
                     f"| Synthetic={len(selected_samples)} | Total={len(self.train_data)}")

        num_examples = train_local_model(self.model, self.train_data)
        logging.info("=== ROUND COMPLETED ===\n")
        return self.get_parameters(config), num_examples, {"client_id": self.client_id}

    def evaluate(self, parameters: List[Any], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        # Dummy evaluation: the server evaluates centrally and skips client evaluation
        return 0.0, len(self.train_data), {}


# === MAIN ===
def main() -> None:
    args = parse_args()
    CONFIG["client_id"] = args.client_id
    CONFIG["train_csv_path"] = args.train_csv
    CONFIG["server_address"] = args.server_address
    CONFIG["log_dir"] = args.log_dir
    CONFIG["client_log_dir"] = os.path.join(args.log_dir, "clients", args.client_id)

    log_path = setup_logging(CONFIG["client_log_dir"])
    logging.info("=== CLIENT STARTED ===")
    logging.info(f"CONFIG: {CONFIG}")
    print(f"[CLIENT] {CONFIG['client_id']} | logging to {log_path}")

    print(f"[CLIENT] Loading training data from {CONFIG['train_csv_path']}...")
    train_df = pd.read_csv(CONFIG["train_csv_path"])

    print("[CLIENT] Fitting the TVAE used to restore masked feedback...")
    restore_synthesizer = fit_restore_synthesizer(train_df)

    print(f"[CLIENT] Loading the pre-trained generator from {CONFIG['gen_model_path']}...")
    generator = load_generator(CONFIG["gen_model_path"])

    model = SimpleNN(train_df.drop(columns=[CONFIG["target_column"]]).shape[1])
    client = FLClient(CONFIG["client_id"], model, train_df, restore_synthesizer, generator)

    print(f"[CLIENT] Connecting to {CONFIG['server_address']}...")
    fl.client.start_numpy_client(server_address=CONFIG["server_address"], client=client)


if __name__ == "__main__":
    main()
