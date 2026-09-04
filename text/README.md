# Federated Learning with Server Feedback Project
This project implements a **federated learning (FL)** system using Flower, where:

- The server coordinates model training, evaluates performance on a global test set, and uses Grad-CAM to generate interpretable feedback on misclassified samples.
- The clients train local models on private data and optionally use this feedback to generate synthetic text using Gemma 2B via Hugging Face.

## Set up

1. **Clone the repository:**
```bash
git clone https://github.com/CaoDien2003/flgenai.git
cd flgenai/text
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.8

---

## How to Run

> Make sure to update `"server_address"` inside the `CONFIG` dictionary in `client_textdata.py` and `server_textdata.py`

### Start the Server

```bash
python server_textdata.py --num_rounds 30 --ckpt_every 5
```

- Loads `data/test_server.csv`
- Runs for 30 rounds (configurable)
- Grad-CAM feedback when overfitting or performance plateau is detected

### Start Clients (each in a separate terminal, machine, or run on Google Colab)

```bash
python client_textdata.py
```

Each client will:
- Loads its own CSV dataset
- Trains DistilBERT on local data
- (Optionally) receive SHAP feedback from the server
- Use Gemma 2B model to generate data

---

## Configuration

### In `client_textdata.py`
- Path to the client's training data (`CSV_PATH`)
- Feedback usage (True = use server's Grad-CAM feedback) (`USE_GRADCAM_FEEDBACK`)
- Ratio of synthetic headlines generated without feedback (`LOCAL_GEN_RATIO`)
- Root folder path on Google Drive if using Colab (`DRIVE_ROOT`)
- Directory where data generation without feedback is saved (`LOCAL_ROOT`)
- Directory where feedback and datageneration with feedback are saved (`FEEDBACK_ROOT`)


### In `server_textdata.py`
- Threshold ratio for token importance (relative to max Grad-CAM score)(`REL_IMPORTANCE`)
- Minimum number of tokens to include in feedback (`MIN_TOKENS`)
- Maximum number of tokens to include in feedback (`MAX_TOKENS`)
- Total number of federated learning rounds (`TOTAL_ROUNDS`)
- How often to save server checkpoints (in rounds) (`CKPT_INTERVAL`)
- Directory where feedback is saved (`GC_ROOT`)

---
## Notes

- No raw data is shared between clients and server
- Only model weights and abstracted feedback are exchanged
- Hugging Face authentication is required to run Gemma:
```bash
from huggingface_hub import login
login("your_huggingface_token")
```
---

## Credits

- [Flower](https://flower.dev) - Federated learning framework
- [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert) - Model
- [Gemma](https://huggingface.co/docs/transformers/en/model_doc/gemma) - Data generation

---


## Contact

For inquiries or feedback, feel free to reach out:

- **Email**: phanlekimtinh0987@gmail.com
- **GitHub**: [ktinh02](https://github.com/ktinh02)
