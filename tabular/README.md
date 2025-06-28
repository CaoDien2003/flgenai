# Domain Adaptation of Federated Learning by Data Generation and Server Feedback

This project implements a federated learning (FL) system using [Flower](https://flower.dev/), where:

- The **server** coordinates global training, evaluates model accuracy, and generates feedback using **SHAP** for misclassified samples.
- The **clients** train local models on private data and optionally use server feedback to restore and augment data using **TVAE** (Tabular Variational Autoencoder).

---

## Project Structure

```
.
├── client.py                 # Federated client logic (PyTorch + SDV)
├── server.py                 # Federated server logic (Flower + SHAP)
├── dataset/
│   ├── test_server.csv       # Global test set for evaluation
│   ├── train1.csv            # Client 1 data
│   ├── train2.csv            # Client 2 data
│   └── ...                   # More clients...
├── tvae_gen.pkl              # Pre-trained TVAE generator model
├── requirements.txt          # Combined dependencies (server + client)
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourname/federated-shap-tvae.git
cd federated-shap-tvae
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.8

---

## How to Run

> Make sure to update `"server_address"` inside the `CONFIG` dictionary in `client_tabular.py` and `server_tabular.py`

### Start the Server

```bash
python server_tabular.py
```

- Loads `dataset/test_server.csv`
- Runs for 30 rounds (configurable)
- Evaluates global model and sends SHAP-based feedback when plateau is detected

### Start Clients (each in separate terminal or machine)

```bash
python client_tabular.py
```

Each client will:
- Load its own `trainX.csv`
- Train a local model
- (Optionally) receive SHAP feedback from the server
- Use a local TVAE model to generate and filter synthetic data

> Configure CSV path and `use_feedback` in `CONFIG` inside `client.py`

---

## Features

- SHAP explainability to generate interpretable feedback
- TVAE-based synthetic data generation
- Cosine similarity filtering for high-quality augmentation
- Plateau detection for dynamic feedback
- Modular & easy-to-extend system (e.g. supports SDV, BERT, ResNet)

---

## Configuration

Edit `CONFIG` dictionary in both `server.py` and `client.py` to customize:

- Training rounds
- Feedback threshold
- Batch size, epochs
- Cosine similarity cutoff
- Feedback usage toggle

---

## Logs & Output

- Server logs evaluation metrics to: `server_logs/exp01/summary_log.csv`
- Feedback per round/client is saved to: `server_logs/exp01/feedback_logs/`
- Client logs are saved to: `client_feedback_logs_01/log_*.txt`

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
- **GitHub**: [CaoDien2003](https://github.com/YourGitHubUsername)
- **LinkedIn**: [Điền Cao](https://www.linkedin.com/in/nguyencaodien/)
