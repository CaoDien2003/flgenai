# === IMPORTS ===
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
import logging
from datetime import datetime

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
CONFIG = {
    "server_address": "0.0.0.0:0",  # Address of the Flower server to connect to
    "train_csv_path": path,  # Path to the client's local training CSV file
    "feedback_log_dir": "client_feedback_logs_01",  # Folder to save feedback logs
    "gen_model_path": "tvae_gen.pkl",  # Path to the pre-trained TVAE generator
    "epoch": 50,  # Number of training epochs
    "batch_size": 32,  # Mini-batch size for training
    "gen_multiplier": 3.0,  # Synthetic data size = multiplier × original size
    "cosine_sim_threshold": 0.9,  # Cosine similarity threshold for filtering
    "use_feedback": False,  # True = generate based on server feedback; False = ignore feedback
    "gen_multiplier_fixed": 0.3,  # Synthetic data ratio (used only when use_feedback = False)
}

# Create the log directory if it doesn't exist
os.makedirs(CONFIG["feedback_log_dir"], exist_ok=True)

# === SETUP LOGGING ===
log_filename = os.path.join(CONFIG["feedback_log_dir"], f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s | %(message)s")
logging.info("=== CLIENT STARTED ===")
logging.info(f"CONFIG: {CONFIG}")

# === LOAD LOCAL TRAINING DATA ===
df = pd.read_csv(CONFIG["train_csv_path"])
X = df.drop(columns=["Exited"])
y = df["Exited"]


# === MODEL DEFINITION ===
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


def get_model():
    return SimpleNN(X.shape[1])


# === TRAIN RESTORE TVAE ===
metadata_restore = SingleTableMetadata()
metadata_restore.detect_from_dataframe(df)
tvae_restore = TVAESynthesizer(metadata_restore)
tvae_restore.fit(df)  # This TVAE is used to restore missing values in feedback

# === LOAD GENERATOR TVAE ===
with open(CONFIG["gen_model_path"], "rb") as f:
    tvae_gen = pickle.load(f)  # Pre-trained generator for synthetic data


# === COSINE SIMILARITY FILTERING FUNCTION ===
def match_by_masked_features(gen_df, fb_df, cosine_threshold):
    matched_rows = []
    for i, row in fb_df.iterrows():
        mask = ~row.isna()  # Mask of present (non-missing) features
        if mask.sum() == 0:
            continue
        fb_vec = row[mask].values.reshape(1, -1)
        gen_filtered = gen_df[mask.index[mask]]
        sims = cosine_similarity(gen_filtered.values, fb_vec)[:, 0]
        selected = gen_df[sims >= cosine_threshold]
        matched_rows.extend(selected.to_dict(orient="records"))
    return pd.DataFrame(matched_rows)


# === FLOWER CLIENT DEFINITION ===
class FLClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.train_data = df.copy()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        use_feedback = CONFIG["use_feedback"]
        shap_feedback = json.loads(config.get("feedback", "[]"))

        restored_df = pd.DataFrame()
        selected_samples = pd.DataFrame()

        if use_feedback and shap_feedback:
            logging.info(f"Using feedback: {len(shap_feedback)} samples")
            # Convert feedback into a dataframe
            fb_df = pd.DataFrame([{**x["features"], "Exited": x["label"]} for x in shap_feedback])

            # === RESTORE MISSING VALUES USING TVAE
            restored_df = fb_df.copy()
            for col in restored_df.columns:
                if restored_df[col].isna().sum() > 0:
                    fill_values = tvae_restore.sample(len(restored_df))[col]
                    restored_df[col] = restored_df[col].fillna(fill_values)

            logging.info(
                f"NaN before restore: {fb_df.isna().sum().sum()} | after restore: {restored_df.isna().sum().sum()}")

            # === GENERATE SYNTHETIC DATA
            num_gen = int(CONFIG["gen_multiplier"] * len(self.train_data))
            gen_candidates = tvae_gen.sample(num_gen)
            logging.info(f"Generated {num_gen} samples from TVAE")

            # === FILTER SYNTHETIC SAMPLES SIMILAR TO FEEDBACK
            selected_samples = match_by_masked_features(gen_candidates, restored_df, CONFIG["cosine_sim_threshold"])
            logging.info(
                f"Selected {len(selected_samples)} samples using cosine threshold >= {CONFIG['cosine_sim_threshold']}")

            # === MERGE ALL DATA
            self.train_data = pd.concat([self.train_data, restored_df, selected_samples], ignore_index=True)

        else:
            logging.info("Gen ignore feedback")

            # === GENERATE FIXED SYNTHETIC DATA (no feedback)
            num_gen = int(CONFIG["gen_multiplier_fixed"] * len(df))
            gen_fixed = tvae_gen.sample(num_gen)
            logging.info(f"Generated {num_gen} synthetic samples ignore feedback")

            # === MERGE ALL DATA
            self.train_data = pd.concat([self.train_data, gen_fixed], ignore_index=True)

        # === PREPARE TENSORS FOR TRAINING ===
        X_train = self.train_data.drop(columns=["Exited"])
        y_train = self.train_data["Exited"]
        logging.info(
            f"Final train size: Original={len(df)} | Restored={len(restored_df)} | Synthetic={len(selected_samples)} | Total={len(self.train_data)}")

        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=CONFIG["batch_size"], shuffle=True)

        # === TRAIN LOCAL MODEL ===
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(CONFIG["epoch"]):
            self.model.train()
            total_loss = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bx.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == CONFIG["epoch"] - 1:
                logging.info(f"Epoch {epoch + 1} / {CONFIG['epoch']} - Loss: {avg_loss:.4f}")
                print(f"[CLIENT] Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        logging.info("=== ROUND COMPLETED ===\n")
        return self.get_parameters(config), len(X_tensor), {"client_id": "client1"}

    def evaluate(self, parameters, config):
        # Dummy evaluation (not used)
        return 0.0, len(self.train_data), {}

# === START CLIENT ===
if __name__ == "__main__":
    model = get_model()
    client = FLClient(model)
    fl.client.start_numpy_client(server_address= CONFIG["server_address"], client=client)
