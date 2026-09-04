# Domain Adaptation of Federated Learning by Data Generation and Server Feedback

This project implements a federated learning (FL) system using [Flower](https://flower.dev/), where:

- The **server** coordinates global training, evaluates model accuracy, and generates feedback using **SHAP** for misclassified samples.
- The **clients** train local models on private data and optionally use server feedback to restore and augment data using **TVAE** (Tabular Variational Autoencoder).

---

## Project Structure

```
.
├── client_tabular.py         # Federated client logic (PyTorch + SDV)
├── server_tabular.py         # Federated server logic (Flower + SHAP)
├── dataset/7030/
│   ├── test_server.csv       # Global test set for evaluation
│   ├── train1.csv            # Client 1 data
│   ├── train2.csv            # Client 2 data
│   └── ...                   # train3.csv, train4.csv, train5.csv
├── tvae_gen.pkl              # Pre-trained TVAE generator model
├── logs/                     # All run artefacts, one folder per run (log_dir)
│   └── exp01/
│       ├── server/           # summary_log.csv, summary_time.csv, server_config.json
│       │   └── feedback_logs/    # feedback_round_<N>_client_<id>.json
│       └── clients/
│           └── client1/      # log_<timestamp>.txt (one file per client run)
├── requirements.txt          # Combined dependencies (server + client)
└── README.md                 # Project documentation
```

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/CaoDien2003/flLgenai.git
cd flLgenai/tabular
```

2. **Create and activate a conda environment:**
```bash
conda create -n flgenai-tabular python=3.10 -y
conda activate flgenai-tabular
```

> Requires Python 3.8–3.11 (verified on 3.10.20 via conda). Python ≥ 3.12 is **not** supported — `torch==2.2.2` has no wheel for it.

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## How to Run

> `server_address` defaults to `0.0.0.0:9000` on the server. Edit `CONFIG["server_address"]` in `server_tabular.py` if you need a different port.

### Start the Server

```bash
python server_tabular.py
```

- Loads `dataset/7030/test_server.csv`
- Runs for N rounds (configurable)
- Evaluates global model and sends SHAP-based feedback when plateau is detected

### Start Clients (each in separate terminal or machine)

Run once per client, each with its own `--client-id` and `--train-csv`:

```bash
python client_tabular.py --client-id client1 --train-csv dataset/7030/train1.csv --server-address 127.0.0.1:9000 --log-dir logs/exp01
python client_tabular.py --client-id client2 --train-csv dataset/7030/train2.csv --server-address 127.0.0.1:9000 --log-dir logs/exp01
# ...
```

Each client will:
- Load its own training CSV
- Train a local model
- (Optionally) receive SHAP feedback from the server, keyed to its `--client-id`
- Use a local TVAE model to generate and filter synthetic data

> `--client-id` must be unique per client — the server uses it to keep each client's feedback separate.
> `--log-dir` must match `CONFIG["log_dir"]` on the server so every log of one run lands under the same folder; it defaults to `logs/exp01`. Toggle `use_feedback` in `CONFIG` inside `client_tabular.py`.

---

## Features

- SHAP explainability to generate interpretable feedback
- TVAE-based synthetic data generation
- Cosine similarity filtering for high-quality augmentation
- Plateau detection for dynamic feedback
- Modular & easy-to-extend system (e.g. supports SDV, BERT, ResNet)

---

## Configuration

- `--client-id`, `--train-csv`, `--server-address`, `--log-dir` are passed as CLI flags to `client_tabular.py` (see *How to Run*).
- Everything else is edited in the `CONFIG` dictionary in each script:
  - `server_tabular.py`: `log_dir`, training rounds, plateau window/threshold, SHAP threshold, top-k features
  - `client_tabular.py`: epochs, batch size, learning rate, generation multiplier, cosine similarity cutoff, `use_feedback` toggle
- Point `log_dir` at a new folder for each run you want to keep — re-running with the same one overwrites its CSVs.

---

## Logs & Output

Everything a run produces lives under the run's `log_dir` (`logs/exp01` by default):

| Path | Content |
| --- | --- |
| `logs/exp01/server/summary_log.csv` | One row per round (round 0 = initial parameters): accuracy, loss, precision, recall, f1, confusion matrix |
| `logs/exp01/server/summary_time.csv` | Per-round wall clock: round, aggregation and SHAP feedback time |
| `logs/exp01/server/server_config.json` | The server `CONFIG` of that run plus `total_time_sec` |
| `logs/exp01/server/feedback_logs/feedback_round_<N>_client_<id>.json` | Masked SHAP feedback and the matching real samples sent to that client |
| `logs/exp01/clients/<client-id>/log_<timestamp>.txt` | Client log: restore/generation counts, final train size, per-epoch loss |

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sdv.metadata.metadata'`** (when loading `tvae_gen.pkl`): your installed `sdv` version doesn't match the one `tvae_gen.pkl` was fitted with — SDV's internal module layout changes across releases, so `pickle.load()` needs an exact match. `requirements.txt` pins `sdv==1.22.1` (read from the pickle's embedded `_fitted_sdv_version`); make sure you actually installed that version (`pip show sdv`) rather than a stale one from a previous env.
- **`ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module`** (or `state is not a legacy MT19937 state`): `tvae_gen.pkl` also carries a random state pickled by numpy ≥ 2.0, while `torch==2.2.2` requires numpy < 2. `client_tabular.py` handles this inside `load_generator()`, which patches numpy's pickle reconstructors while the file is read and restores them afterwards — nothing to do, just don't replace that helper with a plain `pickle.load()`.
- **`FileNotFoundError` on `test_server.csv` / `trainX.csv`**: run the scripts from inside `tabular/` (paths in `CONFIG` are relative to that directory), and confirm the file exists under `dataset/7030/`.
- **Client killed with no traceback (`Killed` in the terminal)**: this is the Linux OOM killer, not a bug — check `dmesg | tail -30` for an `Out of memory: Killed process ... python` line. TVAE training is memory-hungry; on WSL2, raise the memory cap in `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=6GB`, then `wsl --shutdown` from PowerShell) and close other processes before retrying.

---

## Example Use Cases

- Data privacy–preserving learning in banks
- Debugging misclassifications using SHAP feedback
- Exploring how synthetic data affects model performance

---

## Notes

- No raw data is shared between clients and server
- Only model weights and abstracted feedback (SHAP masks) are exchanged
- SDV is used locally by clients to generate data

---

## Credits

- [Flower](https://flower.dev) - Federated learning framework
- [SHAP](https://github.com/slundberg/shap) - Model interpretability
- [SDV](https://docs.sdv.dev/sdv) - Synthetic data generation (TVAE)

---


## Contact

For inquiries or feedback, feel free to reach out:

- **Email**: nguyencaodien2003@gmail.com
- **GitHub**: [CaoDien2003](https://github.com/CaoDien2003)
