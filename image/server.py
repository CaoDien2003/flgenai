#!/usr/bin/env python3
"""
Run:
    python server.py --target /data/subset1 --clients 9 --method proposed
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from flwr.common import (FitIns, Parameters, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader


IMG_SIZE = 32
NUM_CLASSES = 10


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# Classifier: ResNet-18  

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # changed 16 -> 64
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    return ResNet(ResNetBlock, [2, 2, 2, 2], num_classes=num_classes)


# Data
class CifarSubsetFolder(ImageFolder):
    """ImageFolder that forces class index = the number i in folder name `class_{i}_{name}`."""

    def find_classes(self, directory):
        classes = sorted(e.name for e in os.scandir(directory) if e.is_dir())
        class_to_idx = {c: int(c.split("_")[1]) for c in classes}
        return classes, class_to_idx


def load_subset(root: str, verbose: bool = True):
    """Load one non-IID subset. ToTensor -> [0,1]"""
    from collections import Counter
    ds = CifarSubsetFolder(root, transform=transforms.Compose([transforms.ToTensor()]))
    if verbose:
        print(f"[data] {root}: {len(ds)} images | class_to_idx = {ds.class_to_idx}")
        print(f"[data] distribution: {dict(sorted(Counter(ds.targets).items()))}")
    return ds

NUM_ROUNDS = 100        # number of communication rounds
N_CLIENTS_FB = 3       # "the three with the lowest accuracy"
N_CLASSES_FB = 3       # "a few classes with the highest error rates"
N_PER_CLASS = 200      # 200 images/class -> 3 x 200 = 600 images/client
TOP_K = 8              # number of Grad-CAM cells sent
TOP_M = 20             # number of misclassified samples, highest confidence
GRID = 8               # 8x8 CAM grid -> 4x4 px patches
GEN_MODE = "refine"
METHODS = ("proposed", "nofeedback", "fedavg", "fedcog")

# FedCOG (github.com/rui-ye/FedCOG)
FEDCOG_START = 50      # --start_round: plain FedAvg before this round
FEDCOG_N = 256         # --synthesis_batch_size
FEDCOG_STEPS = 500     # --g_steps
FEDCOG_LR = 0.01       # --lr_g
FEDCOG_ADV = 0.1       # --adv
FEDCOG_T = 1.0         # --T / --js_T

DEVICE = get_device()
SERVER_LOG_DIR = os.environ.get("FLGENAI_LOG_DIR", "./server_logs")

SERVER_HOST = "0.0.0.0"   
SERVER_PORT = 8081
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"


def grad_cam(model: nn.Module, x: torch.Tensor, y: int, target_layer: nn.Module) -> torch.Tensor:
    """Grad-CAM for one image. x: (3,H,W). Returns CAM (Hf,Wf) normalized to [0,1]."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0))
        logits[0, y].backward()
        a, g = acts["v"][0], grads["v"][0]
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((w * a).sum(0)).detach()
    finally:
        h1.remove(); h2.remove()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def build_feedback_entry(x01: torch.Tensor, y: int, cam: torch.Tensor, sample_id: int,
                         client_prob: float, top_k: int = TOP_K,
                         grid: int = GRID) -> Dict[str, Any]:
    patch = IMG_SIZE // grid
    cam_g = F.interpolate(cam[None, None], size=(grid, grid),
                          mode="bilinear", align_corners=False)[0, 0]
    idx = cam_g.flatten().topk(min(top_k, grid * grid)).indices.tolist()

    positions, values = [], []
    for i in idx:
        r, c = divmod(i, grid)
        positions.append([r, c])
        values.append(x01[:, r * patch:(r + 1) * patch,
                             c * patch:(c + 1) * patch].detach().cpu().numpy().tolist())
    return {"sample_id": int(sample_id), "label": int(y), "grid": grid, "patch": patch,
            "positions": positions, "values": values, "client_prob": float(client_prob)}


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device=None) -> torch.Tensor:
    """prob(W_j(D^t)) — Algorithm 1 line 4."""
    device = device or DEVICE
    model.eval()
    out = [torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader]
    return torch.cat(out)


def select_trustworthy(probs: Dict[str, torch.Tensor], labels: torch.Tensor):

    keys = list(probs)
    stack = torch.stack([probs[k] for k in keys])                    # (M, N, C)
    correct = stack.argmax(2) == labels[None, :]                     # (M, N)
    conf_y = stack.gather(2, labels.view(1, -1, 1).expand(len(keys), -1, 1)).squeeze(2)
    conf_y = conf_y.masked_fill(~correct, -1.0)
    best = conf_y.argmax(0)
    return [keys[i] for i in best.tolist()], correct.any(0)


def per_class_error(pred: torch.Tensor, labels: torch.Tensor) -> Dict[int, float]:
    err = {}
    for c in range(NUM_CLASSES):
        m = labels == c
        if int(m.sum()) > 0:
            err[c] = float((pred[m] != c).float().mean())
    return err


def algorithm1(global_model: nn.Module, local_models: Dict[str, nn.Module],
               test_set, test_loader: DataLoader, target_layer_of,
               chosen_clients: List[str], n_classes: int = N_CLASSES_FB,
               top_k: int = TOP_K, top_m: int = TOP_M, grid: int = GRID,
               device=None, names: Dict[str, str] = None) -> Dict[str, List[Dict[str, Any]]]:

    device = device or DEVICE
    names = names or {}
    labels = torch.tensor([int(test_set[i][1]) for i in range(len(test_set))])

    probs = {"__global__": predict_probs(global_model, test_loader, device)}     # lines 3-5
    for cid, m in local_models.items():
        probs[cid] = predict_probs(m, test_loader, device)

    best_of, has_correct = select_trustworthy(probs, labels)                     # line 9
    models = {"__global__": global_model, **local_models}

    feedback: Dict[str, List[Dict[str, Any]]] = {}
    for cid in chosen_clients:
        p = probs[cid]
        pred, conf = p.argmax(1), p.max(1).values
        wrong = pred != labels                                                   # line 13

        eligible = wrong & has_correct
        avail = set(labels[eligible].tolist())
        err = per_class_error(pred, labels)
        top_cls = sorted([c for c in err if c in avail],
                         key=err.get, reverse=True)[:n_classes]
        if len(top_cls) < n_classes:
            print(f"    [{names.get(cid, cid)}] only {len(top_cls)}/{n_classes} classes feasible "
                  f"(no model ever predicted the other classes correctly)")

        quota = max(1, top_m // max(len(top_cls), 1))
        cand, spare = [], []
        for c in top_cls:
            pool = [(i, float(conf[i])) for i in range(len(labels))              
                    if wrong[i] and int(labels[i]) == c and has_correct[i]]
            pool.sort(key=lambda t: -t[1])                                      
            cand.extend(pool[:quota])
            spare.extend(pool[quota:])
        spare.sort(key=lambda t: -t[1])
        cand.extend(spare[:max(0, top_m - len(cand))])                          
        cand.sort(key=lambda t: -t[1])

        entries = []
        for sid, cf in cand:
            model = models[best_of[sid]]
            x, y = test_set[sid]
            cam = grad_cam(model, x.to(device), int(y), target_layer_of(model))  
            entries.append(build_feedback_entry(x, int(y), cam, sid, cf, top_k, grid))
        feedback[cid] = entries                                                 

        covered = sorted({e["label"] for e in entries})
        print(f"    [{names.get(cid, cid)}] worst classes {top_cls} "
              f"(err {[round(err[c], 3) for c in top_cls]}) -> {len(entries)} samples "
              f"covering {len(covered)} classes {covered}")
    return feedback


class FeedbackFedAvg(FedAvg):

    def __init__(self, test_set, target_layer_of, *,
                 min_rounds: int = 5,
                 acc_drop_tol: float = 0.0,
                 plateau_patience: int = 3, plateau_delta: float = 2e-3,
                 overfit_gap: float = 0.15,
                 n_clients: int = N_CLIENTS_FB, n_classes: int = N_CLASSES_FB,
                 n_per_class: int = N_PER_CLASS, diff_steps: int = 500,
                 top_k: int = TOP_K,
                 top_m: int = TOP_M, grid: int = GRID, gen_mode: str = GEN_MODE,
                 use_feedback: bool = True, method: str = "proposed",
                 batch_size: int = 128, device=None, **kwargs):
        super().__init__(**kwargs)
        self.test_set     = test_set
        self.test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        self.target_layer_of = target_layer_of
        self.min_rounds   = min_rounds
        self.acc_drop_tol = acc_drop_tol
        self.plateau_patience = plateau_patience
        self.plateau_delta    = plateau_delta
        self.overfit_gap  = overfit_gap
        self.n_clients    = n_clients
        self.n_classes    = n_classes
        self.n_per_class  = n_per_class
        self.diff_steps   = diff_steps
        self.top_k, self.top_m, self.grid = top_k, top_m, grid
        self.gen_mode     = gen_mode
        self.use_feedback = use_feedback   
        self.method       = method        
        self.device       = device or DEVICE

        self.acc_hist: List[float] = []          # global accuracy per round
        self.local_acc: Dict[str, float] = {}    # each client's accuracy on the server test set
        self.local_acc_hist: List[Dict[str, float]] = []   # local_acc per round, by --cid
        self.names: Dict[str, str] = {}          # Flower proxy.cid -> client --cid
        self.pending: Dict[str, List[Dict[str, Any]]] = {}
        self.fb_rounds: List[int] = []           # rounds that actually produced feedback

    def _to_model(self, nds: List[np.ndarray]) -> nn.Module:
        m = resnet18(NUM_CLASSES)
        m.load_state_dict({k: torch.tensor(v)
                           for k, v in zip(m.state_dict().keys(), nds)}, strict=True)
        return m.to(self.device).eval()

    @torch.no_grad()
    def _accuracy(self, model: nn.Module) -> float:
        correct = seen = 0
        model.eval()
        for x, y in self.test_loader:
            pred = model(x.to(self.device)).argmax(1).cpu()
            correct += int((pred == y).sum()); seen += len(y)
        return correct / max(seen, 1)

    def _should_trigger(self, server_round: int, results) -> Tuple[bool, str]:
        if server_round < self.min_rounds:                                 
            return False, f"fewer than {self.min_rounds} rounds"

        h = self.acc_hist
        if len(h) >= 2 and h[-1] < h[-2] - self.acc_drop_tol:               
            return True, f"global dropped {h[-2]:.4f} -> {h[-1]:.4f}"

        if len(h) >= self.plateau_patience + 1:                              
            recent = h[-(self.plateau_patience + 1):]
            span = max(recent) - min(recent)
            if span < self.plateau_delta:
                return True, f"global plateaued for {self.plateau_patience} rounds (span {span:.4f})"

        gaps = {}                                                           
        for proxy, fit_res in results:
            tr = fit_res.metrics.get("train_acc")
            if tr is not None and proxy.cid in self.local_acc:
                gaps[proxy.cid] = float(tr) - self.local_acc[proxy.cid]
        if gaps and max(gaps.values()) > self.overfit_gap:
            worst = max(gaps, key=gaps.get)
            return True, f"overfit: {self.names.get(worst, worst)} train/test gap {gaps[worst]:.3f}"

        return False, "not needed"

    def aggregate_fit(self, server_round: int, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures) 
        if params is None:
            return params, metrics

        global_model = self._to_model(parameters_to_ndarrays(params))
        local_models = {proxy.cid: self._to_model(parameters_to_ndarrays(fr.parameters))
                        for proxy, fr in results}
        self.names.update({proxy.cid: str(fr.metrics.get("client_id", proxy.cid))
                           for proxy, fr in results})
        name = lambda cid: self.names.get(cid, cid)

        acc_g = self._accuracy(global_model)
        self.acc_hist.append(acc_g)
        self.local_acc = {cid: self._accuracy(m) for cid, m in local_models.items()}
        self.local_acc_hist.append({name(c): a for c, a in self.local_acc.items()})
        print(f"[R{server_round}] global acc {acc_g:.4f} | "
              f"client { {name(c): round(a, 3) for c, a in sorted(self.local_acc.items(), key=lambda t: name(t[0]))} }")

        if self.method in ("fedavg", "fedcog"):
            return params, metrics            # baselines: no server-side generation

        trigger, why = self._should_trigger(server_round, results)            
        if not trigger:
            print(f"[R{server_round}] no feedback generated ({why})")
            return params, metrics

        if self.use_feedback:
            chosen = sorted(self.local_acc, key=self.local_acc.get)[:self.n_clients]  
            print(f"[R{server_round}] GENERATING FEEDBACK ({why}) -> {len(chosen)} lowest "
                  f"clients: {[name(c) for c in chosen]}")
            self.pending = algorithm1(global_model, local_models, self.test_set,
                                      self.test_loader, self.target_layer_of, chosen,
                                      n_classes=self.n_classes, top_k=self.top_k,
                                      top_m=self.top_m, grid=self.grid, device=self.device,
                                      names=self.names)
        else:
            rng = random.Random(server_round)
            chosen = rng.sample(sorted(self.local_acc), min(self.n_clients, len(self.local_acc)))
            print(f"[R{server_round}] GENERATING DATA, no feedback ({why}) -> {len(chosen)} "
                  f"random clients: {[name(c) for c in chosen]}")
            self.pending = {cid: {"__gen_budget__": {"n_classes": self.n_classes,
                                                     "n_per_class": self.n_per_class}}
                            for cid in chosen}

        produced = sum(len(v) for v in self.pending.values())
        if produced:
            self.fb_rounds.append(server_round)
        else:
            print(f"[R{server_round}] no eligible samples -> not counted as a data generation round")
            self.pending = {}

        if self.use_feedback:
            os.makedirs(SERVER_LOG_DIR, exist_ok=True)
            with open(os.path.join(SERVER_LOG_DIR,
                                   f"feedback_round_{server_round}.json"), "w") as f:
                json.dump({c: [{k: v for k, v in e.items() if k != "values"} for e in fbs]
                           for c, fbs in self.pending.items()}, f, indent=2)
        return params, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        if not self.pending:
            return ins

        out = []
        for proxy, fit_ins in ins:
            cfg = dict(fit_ins.config)
            fbs = self.pending.get(proxy.cid)
            if fbs and not self.use_feedback:
                b = fbs["__gen_budget__"]          
                cfg.update({"gen_budget": json.dumps(b), "diff_steps": self.diff_steps})
                print(f"[R{server_round}] -> {self.names.get(proxy.cid, proxy.cid)}: generate "
                      f"{b['n_classes']} x {b['n_per_class']} images, classes chosen by the "
                      f"client (no feedback)")
            elif fbs:
                cfg.update({"feedback": json.dumps(fbs), "n_per_class": self.n_per_class,
                            "mode": self.gen_mode, "top_m": self.top_m,
                            "diff_steps": self.diff_steps})
                print(f"[R{server_round}] -> {self.names.get(proxy.cid, proxy.cid)}: {len(fbs)} feedback elements "
                      f"({len(json.dumps(fbs)) / 1024:.0f} KB)")
            out.append((proxy, FitIns(fit_ins.parameters, cfg)))

        self.pending = {}      # feedback is single-use
        return out


def weighted_average(metrics):
    total = sum(n for n, _ in metrics) or 1
    keys = {k for _, m in metrics for k, v in m.items() if isinstance(v, (int, float))}
    return {k: sum(n * m[k] for n, m in metrics if k in m) / total for k in keys}


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FLGenAI server")
    p.add_argument("--target", required=True,
                   help="subset directory used as target set D^t ")
    p.add_argument("--clients", type=int, default=9,
                   help="minimum number of clients (10 subsets, 1 as target -> 9)")
    p.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    # §2 — trigger conditions
    p.add_argument("--min-rounds", type=int, default=5)
    p.add_argument("--acc-drop-tol", type=float, default=0.0)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--plateau-delta", type=float, default=2e-3)
    p.add_argument("--overfit-gap", type=float, default=0.15)
    # §3.4 — budget
    p.add_argument("--n-clients", type=int, default=N_CLIENTS_FB)
    p.add_argument("--n-classes", type=int, default=N_CLASSES_FB)
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--diff-steps", type=int, default=500,
                   help="number of reverse diffusion steps on the client")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--top-m", type=int, default=TOP_M)
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--mode", default=GEN_MODE, choices=["refine", "repaint", "paper"],
                   help="image generation mode forced on clients")
    p.add_argument("--no-feedback", action="store_true",
                   help="generate images without send feedback")
    p.add_argument("--method", default="proposed", choices=METHODS,
                   help="proposed | nofeedback | fedavg | fedcog")
    # FedCOG (only used with --method fedcog)
    p.add_argument("--fedcog-start", type=int, default=FEDCOG_START)
    p.add_argument("--fedcog-n", type=int, default=FEDCOG_N)
    p.add_argument("--fedcog-steps", type=int, default=FEDCOG_STEPS)
    p.add_argument("--fedcog-lr", type=float, default=FEDCOG_LR)
    p.add_argument("--fedcog-adv", type=float, default=FEDCOG_ADV)
    p.add_argument("--fedcog-T", type=float, default=FEDCOG_T)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-dir", default=SERVER_LOG_DIR)
    return p.parse_args(argv)


def main(argv=None):
    global DEVICE, SERVER_LOG_DIR

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(argv)
    if args.no_feedback:
        args.method = "nofeedback"
    set_seed(args.seed)
    DEVICE = get_device(args.device)
    SERVER_LOG_DIR = args.log_dir
    os.makedirs(SERVER_LOG_DIR, exist_ok=True)
    print(f"[server] device={DEVICE} rounds={args.rounds} clients={args.clients} "
          f"method={args.method}")

    def fit_config(server_round: int):
        """Sent to every client each round; tells it which method to run."""
        cfg = {"method": args.method, "server_round": server_round}
        if args.method == "fedcog":
            cfg.update({"fedcog_start": args.fedcog_start, "fedcog_n": args.fedcog_n,
                        "fedcog_steps": args.fedcog_steps, "fedcog_lr": args.fedcog_lr,
                        "fedcog_adv": args.fedcog_adv, "fedcog_T": args.fedcog_T})
        return cfg

    target_set = load_subset(args.target)

    init = resnet18(NUM_CLASSES)
    initial_parameters = ndarrays_to_parameters(
        [v.cpu().numpy() for v in init.state_dict().values()])

    strategy = FeedbackFedAvg(
        test_set=target_set,
        target_layer_of=lambda m: m.layer3,   # 8x8 feature map -> 8x8 CAM grid
        initial_parameters=initial_parameters,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=args.clients, min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        min_rounds=args.min_rounds, acc_drop_tol=args.acc_drop_tol,
        plateau_patience=args.plateau_patience, plateau_delta=args.plateau_delta,
        overfit_gap=args.overfit_gap,
        n_clients=args.n_clients, n_classes=args.n_classes,
        n_per_class=args.n_per_class, diff_steps=args.diff_steps,
        top_k=args.top_k, top_m=args.top_m,
        grid=args.grid, gen_mode=args.mode,
        use_feedback=args.method != "nofeedback", method=args.method,
        on_fit_config_fn=fit_config,
        device=DEVICE,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"[server] listening on {SERVER_ADDRESS} ...")
    hist = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print(f"\n[server] data generation rounds: {strategy.fb_rounds}")
    print(f"[server] ({len(strategy.fb_rounds)}/{args.rounds} rounds — paper: 16-18/50)")
    with open(os.path.join(SERVER_LOG_DIR, "summary.json"), "w") as f:
        json.dump({"method": args.method,
                   "acc_hist": strategy.acc_hist,                 # global model on D^t
                   "local_acc_hist": strategy.local_acc_hist,     # each local model on D^t
                   "fb_rounds": strategy.fb_rounds,
                   "client_val_loss": hist.losses_distributed,    # weighted mean of client val
                   "client_val_acc": hist.metrics_distributed.get("accuracy", []),
                   "client_train": hist.metrics_distributed_fit}, f, indent=2)


if __name__ == "__main__":
    main()
