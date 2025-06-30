
import argparse, json, glob, os, warnings
from typing import Dict, List

import flwr as fl
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

P = argparse.ArgumentParser()
P.add_argument("--num_rounds", type=int, default=100)
P.add_argument("--ckpt_every", type=int, default=1,  
               help="Save a server checkpoint every N rounds")
ARGS          = P.parse_args()
TOTAL_ROUNDS  = ARGS.num_rounds
CKPT_INTERVAL = max(1, ARGS.ckpt_every)
DEVICE_GC = torch.device("cpu") 

REL_IMPORTANCE = 0.35   # keep tokens whose score ≥ 35 % of the max
MIN_TOKENS     = 3      
MAX_TOKENS     = 10


GC_ROOT = "gc_runs"
os.makedirs(GC_ROOT, exist_ok=True)

def save_gc_run(round_idx: int,
                send_to_clients: Dict[int, list],
                per_sample: list[dict]):
    out_dir = os.path.join(GC_ROOT, f"round_{round_idx}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "heatmaps_to_clients.json"), "w",
              encoding="utf-8") as f:
        json.dump(send_to_clients, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "per_sample.jsonl"), "w",
              encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Server]   Grad-CAM run saved → {out_dir}")

# data test
TEST_CSV = "test.csv"
df_test  = pd.read_csv(TEST_CSV)
lbl2id   = {l: i for i, l in enumerate(sorted(df_test.label.unique()))}
df_test["label"] = df_test.label.map(lbl2id)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_test = tokenizer(
    df_test.text.tolist(),
    truncation=True, padding="max_length", max_length=64, return_tensors="pt"
)
TEST_LOADER = DataLoader(
    TensorDataset(
        enc_test["input_ids"],
        enc_test["attention_mask"],
        torch.tensor(df_test.label.tolist()),
    ),
    batch_size=64,
)

# model
class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels: int = 42):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

def get_parameters(m) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for p in m.state_dict().values()]

def set_parameters(m, parameters: List[np.ndarray]) -> None:
    sd = m.state_dict()
    for (k, _), p in zip(sd.items(), parameters):
        sd[k] = torch.tensor(p, dtype=sd[k].dtype)
    m.load_state_dict(sd, strict=True)

# checkpoint
CKPT_DIR = "server_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

def ckpt_path(r: int) -> str:
    return os.path.join(CKPT_DIR, f"round_{r}.pt")

def save_ckpt(parameters_nd, server_round: int, hist: list[dict]):
    torch.save(
        {"round": server_round,
         "parameters": parameters_nd,
         "hist": hist},
        ckpt_path(server_round)
    )
    print(f"[Server]   saved checkpoint → {ckpt_path(server_round)}")

def load_latest_ckpt():
    """Return checkpoint dict or None; tolerate old unsafe pickles."""
    ckpt_files = glob.glob(os.path.join(CKPT_DIR, "round_*.pt"))
    if not ckpt_files:
        return None

    ckpt_files = sorted(
        ckpt_files,
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    latest = ckpt_files[-1]

    try:
        ck = torch.load(latest, map_location="cpu")          
    except (pickle.UnpicklingError, RuntimeError):
        print(f"[Server]  {latest} uses legacy pickle – loading unsafely")
        ck = torch.load(latest, map_location="cpu", weights_only=False)

    print(f"[Server]  loaded checkpoint {latest}")
    return ck

# GradCAM
def _position(i, n):
    return "beginning" if i / n < .33 else "middle" if i / n < .66 else "end"

def _gradcam_one_sample(model, text, gold, top_k=4):
    model.eval()
    dev = next(model.parameters()).device
    inp = tokenizer(text, truncation=True, padding="max_length",
                    max_length=64, return_tensors="pt").to(dev)
    # inp = tokenizer(text, truncation=True, padding="max_length",
    #                 max_length=64, return_tensors="pt").to(DEVICE)
    emb_out = None
    def hook(_, __, out):
        nonlocal emb_out
        emb_out = out
    h = model.model.distilbert.embeddings.register_forward_hook(hook)
    logits = model(**inp).logits
    pred   = logits.argmax(1).item()
    if emb_out is None:
        h.remove()
        return []
    emb_out.retain_grad()
    model.zero_grad()
    logits[0, pred].backward(retain_graph=True)
    grads = emb_out.grad.squeeze(0).norm(dim=1)
    toks  = tokenizer.convert_ids_to_tokens(inp["input_ids"].squeeze(0))
    ignore = {"[CLS]", "[SEP]", "[PAD]"}
    scored = [(i, tk, grads[i].item())
              for i, tk in enumerate(toks) if tk not in ignore]


    if not scored:
        h.remove(); return []

    top_score   = max(s for *_ , s in scored)
    threshold   = max(REL_IMPORTANCE * top_score, 1e-9)
    picked      = [t for t in scored if t[2] >= threshold]

    # enforce sensible min/max lengths
    picked.sort(key=lambda x: x[2], reverse=True)
    if len(picked) < MIN_TOKENS:
        picked = picked + sorted(
            [t for t in scored if t not in picked],
            key=lambda x: x[2], reverse=True
        )[: MIN_TOKENS - len(picked)]
    picked = picked[:MAX_TOKENS]

    h.remove()
    return [(tk, _position(i, len(toks)), score) for i, tk, score in picked]

def _probs_and_preds(model):
    model.eval()
    dev = next(model.parameters()).device
    probs_all, preds_all = [], []
    with torch.no_grad():
        for ids, msk, _ in TEST_LOADER:
            ids, msk = ids.to(dev), msk.to(dev)
            logits   = model(ids, attention_mask=msk).logits
            p        = softmax(logits, dim=1).cpu().numpy()
            probs_all.append(p)
            preds_all.append(p.argmax(1))
    return np.vstack(probs_all), np.concatenate(preds_all)

def _best_model_per_sample(probs_stack, preds_stack, gold):
    n = gold.shape[0]
    best_idx = [-1] * n
    best_p   = np.zeros(n)
    for j, (pr, pd) in enumerate(zip(probs_stack, preds_stack)):
        choose = (pd == gold) & (pr[np.arange(n), gold] > best_p)
        best_p[choose] = pr[choose, gold[choose]]
        for i in np.where(choose)[0]:
            best_idx[i] = j
    return best_idx

# evaluation callback 
def server_eval(server_round, par, _):
    set_parameters(
        GLOBAL_MODEL,
        parameters_to_ndarrays(par) if hasattr(par, "tensors") else par
    )
    GLOBAL_MODEL.eval()
    corr = tot = 0
    with torch.no_grad():
        for ids, msk, lbl in TEST_LOADER:
            ids, msk, lbl = ids.to(DEVICE), msk.to(DEVICE), lbl.to(DEVICE)
            corr += (
                GLOBAL_MODEL(ids, attention_mask=msk).logits.argmax(1) == lbl
            ).sum().item()
            tot += lbl.size(0)
    acc = corr / tot if tot else 0.0
    print(f"[Server]  Round {server_round} – test acc {acc:.4f}")
    return 0.0, {"server_acc": acc}


class MyFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, start_round: int, **kw):
        super().__init__(**kw)
        self.start_round      = start_round
        self.hist: list[dict] = []
        self.mis_prev: Dict[int, list] = {}

        self.gc_has_fired = False
        self._current_parameters = kw.get("initial_parameters")
        self.overfit_threshold = 0.95
        self.plateau_patience  = 3
        self._save_ckpt = save_ckpt
        self.global_round_done = max(start_round - 1, 0)

    def should_run_gc(self) -> str | None:

        # if self.gc_has_fired:
        #     return "forced"
        
        if not self.hist:
            return None
        if self.hist[-1]["train"] > self.overfit_threshold:
            return "overfit"
        if len(self.hist) >= self.plateau_patience + 1:
            last = [h["test"] for h in self.hist[-(self.plateau_patience+1):]]
            if max(last) - min(last) < 0.01:
                return "plateau"
        return None

    def aggregate_fit(self, server_round, results, failures):
        self.global_round_done += 1
        global_r = self.global_round_done 

        if not results:
            warnings.warn("No client returned; skipping round.")
            return self._current_parameters, {}

        agg = super().aggregate_fit(server_round, results, failures)
        params_nd, _ = agg
        self._current_parameters = params_nd

        train_acc = np.mean([r.metrics.get("train_accuracy", 0) for _, r in results])
        test_acc  = server_eval(global_r, params_nd, {})[1]["server_acc"]
        self.hist.append({"train": train_acc, "test": test_acc})


        reason = self.should_run_gc()
        if reason:
            print(f"[Server]   Grad-CAM triggered at ROUND {server_round} ({reason})")

            gold = df_test.label.values
            probs_stack, preds_stack, model_tags, param_blobs = [], [], [], []

            # global model
            mdl = DistilBertClassifier(num_labels=42).to(DEVICE_GC)
            set_parameters(mdl, parameters_to_ndarrays(params_nd))
            p, y = _probs_and_preds(mdl)           
            probs_stack.append(p); preds_stack.append(y)
            model_tags.append("global")
            param_blobs.append(parameters_to_ndarrays(params_nd))  
            del mdl                                              ; torch.cuda.empty_cache()

            # each client model 
            client_ids = [] 
            for idx, (_, fit_res) in enumerate(results, start=1):
                cid = _.cid
                mdl = DistilBertClassifier(num_labels=42).to(DEVICE_GC)
                par = parameters_to_ndarrays(fit_res.parameters)
                set_parameters(mdl, par)
                p, y = _probs_and_preds(mdl)
                probs_stack.append(p); preds_stack.append(y)
                model_tags.append(f"client_{cid}")
                client_ids.append(cid)
                param_blobs.append(par)                             
                del mdl                                            ; torch.cuda.empty_cache()

            # best model per sample 
            client_ids   = [cp.cid for cp, _ in results]      
            GC_SCORES    = []      # tuples: (gc, idx, lbl, heat, prob, cid)
            err_counts   = defaultdict(lambda: defaultdict(int))  # class → cid → wrong

            # loop over each client model (skip index 0 which is the global model)
            for j, cid in enumerate(client_ids, start=1):
                wrong_idx = np.where(preds_stack[j] != gold)[0]     # this client’s errors
                if wrong_idx.size == 0:
                    continue

                mdl = DistilBertClassifier(num_labels=42).to(DEVICE_GC)
                set_parameters(mdl, param_blobs[j])

                for i in wrong_idx:
                    heat = _gradcam_one_sample(mdl, df_test.text.iloc[i], int(gold[i]))
                    if not heat:
                        continue
                    lbl       = int(gold[i])
                    prob      = probs_stack[j][i, lbl]
                    gc_score  = sum(t[2] for t in heat)

                    GC_SCORES.append((gc_score, i, lbl, heat, prob, cid))
                    err_counts[lbl][cid] += 1

                del mdl; torch.cuda.empty_cache()

            # allocate proportionally 
            GC_SCORES.sort(key=lambda x: (-x[0], -x[4]))   # GC ↑ first, prob ↑ second

            # quota_left = defaultdict(lambda: defaultdict(int))   # class → cid → quota
            # for lbl, cid_dict in err_counts.items():
            #     total_err = sum(cid_dict.values())
            #     for cid, n_err in cid_dict.items():
            #         # proportional quota, at least 1
            #         quota_left[lbl][cid] = max(1, int(len(GC_SCORES) * n_err / len(GC_SCORES)))
            N = len(GC_SCORES)
            quota_left = defaultdict(lambda: defaultdict(int))
            for lbl, cid_dict in err_counts.items():
                total_err_lbl = sum(cid_dict.values())
                for cid, n_err in cid_dict.items():
                    quota_left[lbl][cid] = max(
                        1,
                        int(N * n_err / total_err_lbl)
                    )

            selected = []
            for item in GC_SCORES:
                gc_score, i, lbl, heat, prob, cid = item
                if quota_left[lbl][cid] > 0:
                    selected.append(item)
                    quota_left[lbl][cid] -= 1
                    # optional early-stop if all quotas finished:
                    # if not any(v for d in quota_left.values() for v in d.values()):
                    #     break

            # build per-client feedback dict 
            feedback_by_client = defaultdict(lambda: defaultdict(list))
            for gc_score, i, lbl, heat, prob, cid in selected:
                feedback_by_client[cid][lbl].append({"heatmap": heat})            
                
                # save_gc_run(global_r, feedback_by_client, [])
                # self.mis_prev = feedback_by_client

            # self.gc_has_fired = True 


        if global_r % CKPT_INTERVAL == 0:
            save_ckpt(params_nd, global_r, self.hist)

        return agg


    # send GC feedback next round 
    def configure_fit(self, server_round, parameters, client_manager, **kw):
        ins = super().configure_fit(
            server_round, parameters, client_manager, **kw
        )
        for client_proxy, fit_ins in ins:
            cid = client_proxy.cid
            fit_ins.config["server_round"] = server_round
            if cid in self.mis_prev:
                fit_ins.config["misclass_map"] = json.dumps(self.mis_prev[cid])
        self.mis_prev = {}
        return ins


GLOBAL_MODEL = DistilBertClassifier(num_labels=42).to(DEVICE)
ck = load_latest_ckpt()
if ck:
    par_nd = (parameters_to_ndarrays(ck["parameters"])
              if hasattr(ck["parameters"], "tensors") else ck["parameters"])

    set_parameters(GLOBAL_MODEL, par_nd)
    start_round = ck["round"] + 1
    hist_init   = ck["hist"]
    _ = server_eval(start_round - 1, par_nd, {})    
else:
    start_round = 0
    hist_init   = []

strategy = MyFedAvg(
    start_round=start_round,
    evaluate_fn=server_eval,
    initial_parameters=ndarrays_to_parameters(get_parameters(GLOBAL_MODEL)),
    fraction_fit=1.0, fraction_evaluate=1.0,
    min_fit_clients=5, min_available_clients=5, min_evaluate_clients=5,
)
strategy.hist.extend(hist_init)
ck_round = ck["round"] if ck else 0
strategy.global_round_done = ck_round

print(f"[Server] starting from round {start_round} "
      f"(running {TOTAL_ROUNDS - start_round} more rounds)… "
      f"– checkpoint every {CKPT_INTERVAL} rounds")


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS - start_round),
    strategy=strategy,
)
