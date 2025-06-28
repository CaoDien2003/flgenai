import flwr as fl
import json, os, glob, random, pathlib
from collections import Counter
from typing import List, Dict

import re
import numpy as np
import pandas as pd
import torch, gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    pipeline,
    AutoModelForCausalLM
)
from functools import lru_cache

from huggingface_hub import login
login("******") 

MODEL_ID = "google/gemma-2b-it" 
dtype    = torch.float16

CSV_PATH      = "/content/drive/MyDrive/Colab Notebooks/CS438/client4.csv"
CKPT_DIR      = "/content/drive/MyDrive/Colab Notebooks/CS438/client_ckpt"

DRIVE_ROOT    = "/content/drive/MyDrive/Colab Notebooks/CS438"
FEEDBACK_ROOT = f"{DRIVE_ROOT}/datagen_feedback"
LOCAL_ROOT    = f"{DRIVE_ROOT}/datagen_local"

USE_GRADCAM_FEEDBACK = True        # set False for experiment-2
LOCAL_GEN_RATIO      = 0.5         # used when no feedback
BALANCE_DATA         = True     

os.makedirs(CKPT_DIR, exist_ok=True)

def ckpt_path(r: int) -> str:
    return os.path.join(CKPT_DIR, f"round_{r}.pt")


def load_latest_ckpt(model: nn.Module):
    """return (last_round, meta) or (-1, {}) if no checkpoint."""
    ckpt_files = glob.glob(os.path.join(CKPT_DIR, "round_*.pt"))
    if not ckpt_files:
        return -1, {}

    ckpt_files = sorted(
        ckpt_files,
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    latest = ckpt_files[-1]
    ck = torch.load(latest, map_location="cpu")
    model.load_state_dict(ck["model"])
    print(f"[Client]  resumed from {latest}")
    return ck["round"], ck.get("meta", {})


def save_ckpt(model: nn.Module, r: int, meta: dict):
    """persist model weights for round r"""
    torch.save(
        {"round": r, "model": model.state_dict(), "meta": meta},
        ckpt_path(r)
    )
    print(f"[Client]   saved ckpt → {ckpt_path(r)}")

class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels=42):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

REL_IMPORTANCE = 0.35
tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=True,)
df_orig    = pd.read_csv(CSV_PATH)
label2id   = {s: i for i, s in enumerate(sorted(df_orig.label.unique()))}
LABEL_NAMES = {v: k for k, v in label2id.items()}
df_orig["label"] = df_orig.label.map(label2id)

texts  : List[str] = df_orig.text.tolist()
labels : List[int] = df_orig.label.tolist()

@lru_cache(maxsize=None)
def get_gemma_pipeline(batch_size: int = 8):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0},          # all weights ➜ GPU-0
        # device_map="cpu",
        low_cpu_mem_usage=True,
    )
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=60,
        top_p=0.9,
        return_full_text=False,
        batch_size=batch_size,
    )

GEMMA = get_gemma_pipeline()

gc.collect(); torch.cuda.empty_cache()

def _clean_tok(t: str) -> str:
    return t.lstrip("#").replace("▁", "").strip("ĠĊ")

def build_prompt(heat, label_name):
    loc = {"beginning": "at the beginning",
           "middle":    "in the middle",
           "end":       "at the end"}
    words = [f"'{_clean_tok(tk)}' {loc[pos]}" for tk, pos, *_ in heat]
    want  = ", ".join(words)
    return (f"Write one realistic news headline of around thirty words for '{label_name}' "
            f"that contains {want}. Natural language only:")


def gen_headline(prompt: str, must: list[str]) -> str | None:
    gemma = get_gemma_pipeline()            # cheap (cached)
    txt = gemma(prompt, temperature=0.7)[0]["generated_text"].strip()

    if len(txt.split()) < 25:               # 25–40 words
        return None
    if sum(m in txt.lower() for m in must) < max(1, len(must) // 2):
        return None
    return " ".join(txt.split()[:40])

def _load_past_synthetic(upto: int):
    """merge every json headline ≤ previous round into texts & labels"""
    for root in (FEEDBACK_ROOT, LOCAL_ROOT):
        for rdir in pathlib.Path(root).glob("round_*"):
            try:
                rid = int(rdir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if rid > upto:
                continue
            for j in rdir.glob("class_*/headlines.json"):
                cls = int(j.parent.name.split("_")[1])
                with open(j, encoding="utf-8") as f:
                    lst = json.load(f)
                texts.extend(lst)
                labels.extend([cls] * len(lst))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build model & resume (if any)
model  = DistilBertClassifier(num_labels=42).to(device)
last_round_done, _ = load_latest_ckpt(model)      # -1 if none
_load_past_synthetic(last_round_done)             # restore json data
print(f"[Client]  dataset size after merge: {len(texts)}")

# initial tokenisation
enc_inp = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=64, return_tensors="pt")

class NewsDataset(Dataset):
    def __len__(self):
        return len(labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      enc_inp["input_ids"][idx],
            "attention_mask": enc_inp["attention_mask"][idx],
            "labels":         torch.tensor(labels[idx]),
        }

def _make_dataloaders():
    ds_full = NewsDataset()
    tr_len  = int(0.8 * len(ds_full))
    train_ds, test_ds = random_split(
        ds_full, [tr_len, len(ds_full) - tr_len],
        generator=torch.Generator().manual_seed(42)
    )

    # class-imbalanced –> weighted sampler
    if BALANCE_DATA:
        cnt = Counter([labels[i] for i in train_ds.indices])
        weights = torch.tensor([1.0 / cnt[l] for l in labels], dtype=torch.float)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights[train_ds.indices], len(train_ds.indices), replacement=True
        )
        train_dl_ = DataLoader(train_ds, batch_size=32, sampler=sampler,
                               pin_memory=True)
    else:
        train_dl_ = DataLoader(train_ds, batch_size=32, shuffle=True,
                               pin_memory=True)

    test_dl_ = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True)
    return train_dl_, test_dl_

train_dl, test_dl = _make_dataloaders()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

class NewsClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        # start with last finished round so that first fit() will align
        self.offset = max(0, last_round_done)
        self.round = last_round_done


    def get_parameters(self):
        return [p.detach().cpu().numpy() for p in model.state_dict().values()]

    def set_parameters(self, p):
        sd = model.state_dict()
        for (k, _), param in zip(sd.items(), p):
            sd[k] = torch.tensor(param, dtype=sd[k].dtype)
        model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        global enc_inp, train_dl, test_dl, texts, labels

        srv_round   = int(config.get("server_round", 1))      # Flower’s session round
        self.round  = self.offset + srv_round                 # global round index
        print(f"[Client]  >>> GLOBAL ROUND {self.round} (server round {srv_round})")

        self.set_parameters(parameters)

        # data generation
        prev_len = len(texts)

        # Datagen with feedback
        if config.get("misclass_map") and USE_GRADCAM_FEEDBACK:
            mis = json.loads(config["misclass_map"])
            base = os.path.join(FEEDBACK_ROOT, f"round_{self.round}")
            for cls_s, lst in mis.items():
                cls  = int(cls_s)
                cdir = os.path.join(base, f"class_{cls}")
                os.makedirs(cdir, exist_ok=True)
                with open(os.path.join(cdir, "heatmaps.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(lst, f, ensure_ascii=False, indent=2)

                gen = []
                TARGET_PER_HEAT = 1         
                MAX_TRIES       = 10    

                for item in lst:
                    heat   = item["heatmap"]
                    prompt = build_prompt(heat, LABEL_NAMES[cls])
                    must   = [_clean_tok(tk).lower() for tk, *_ in heat]

                    prompts   = [prompt]*MAX_TRIES          # create a mini-batch

                    gemma = get_gemma_pipeline()            # cached
                    outs = GEMMA(prompts, temperature=0.8)      # one scalar → valid

                    good = 0
                    for out_list in outs:                       # each element is a list
                        text = out_list[0]["generated_text"].strip()

                        if len(text.split()) < 25:
                            continue
                        if sum(m in text.lower() for m in must) < max(1, len(must)//2):
                            continue
                        if text in gen:
                            continue

                        gen.append(text)
                        texts.append(text)
                        labels.append(cls)
                        good += 1
                        if good >= TARGET_PER_HEAT:
                            break

                with open(os.path.join(cdir, "headlines.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(gen, f, ensure_ascii=False, indent=2)

        # Datagen without feedback
        elif not USE_GRADCAM_FEEDBACK:
            n_new = int(round(len(texts) * LOCAL_GEN_RATIO))
            base  = os.path.join(LOCAL_ROOT, f"round_{self.round}")
            os.makedirs(base, exist_ok=True)
            by_cls: Dict[int, List[str]] = {}
            for _ in range(n_new):
                cls = random.choice(labels)
                hl  = gen_headline(
                    "Write one realistic 30-word news headline:", [])
                by_cls.setdefault(cls, []).append(hl)
                texts.append(hl)
                labels.append(cls)
            for cls, lst in by_cls.items():
                cdir = os.path.join(base, f"class_{cls}")
                os.makedirs(cdir, exist_ok=True)
                with open(os.path.join(cdir, "headlines.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(lst, f, ensure_ascii=False, indent=2)

        # Re-tokenise & build loaders if new data added
        if len(texts) > prev_len:
            enc_inp = tokenizer(texts, truncation=True, padding="max_length",
                                max_length=64, return_tensors="pt")
            train_dl, test_dl = _make_dataloaders()

            pd.DataFrame({"text": texts, "label": labels}).to_csv(
                f"{DRIVE_ROOT}/all_data_round{self.round}.csv", index=False
            )
            print(f"[Client]   CSV saved for round {self.round} "
                  f"({len(texts)} rows)")

        model.train()
        opt = optim.AdamW(model.parameters(), lr=2e-5)

        NUM_CLASSES = 42                

        if BALANCE_DATA:
            cnt = Counter(labels)                   
            w   = torch.ones(NUM_CLASSES,           
                            dtype=torch.float)

            for lbl, freq in cnt.items():          
                if lbl < NUM_CLASSES:               
                    w[lbl] = 1.0 / freq             

            loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        for _ in range(3):                      # 3 epochs
            for b in train_dl:
                ids = b["input_ids"].to(device)
                msk = b["attention_mask"].to(device)
                lbl = b["labels"].to(device)
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(ids, attention_mask=msk).logits
                    loss   = loss_fn(logits, lbl)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        save_ckpt(model, self.round, {})

        return self.get_parameters(), len(train_dl.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for b in test_dl:
                ids = b["input_ids"].to(device)
                msk = b["attention_mask"].to(device)
                lbl = b["labels"].to(device)
                corr += (model(ids, attention_mask=msk).logits.argmax(1)
                         == lbl).sum().item()
                tot  += lbl.size(0)
        return 0.0, tot, {"accuracy": corr / tot if tot else 0.0}

fl.client.start_numpy_client(
    server_address="128.214.252.95:8080",
    client=NewsClient(),
)
