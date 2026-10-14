#!/usr/bin/env python3
"""Client 
The method (proposed / nofeedback / fedavg / fedcog) is chosen on the server
Run:
    python client.py --data /data/subset5 --cid client_4
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import random
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from flwr.common import Scalar
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image
from tqdm import trange

IMG_SIZE = 32
NUM_CLASSES = 10


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_diff(x01):   return x01 * 2.0 - 1.0


def from_diff(xd):  return ((xd + 1.0) / 2.0).clamp(0.0, 1.0)

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


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(   # 32x32
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )

    def forward(self, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        return self.net(torch.cat([input, class_embed, timestep_embed], dim=1))


def get_alphas_sigmas(log_snrs):
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    return -torch.special.expm1(1e-4 + 10 * t**2).log()


@torch.no_grad()
def sample(model, x, steps, eta, classes):
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * log_snrs[i], classes).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    return pred


def load_diffusion(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "model_ema" if "model_ema" in ckpt else "model"
    model = Diffusion()
    model.load_state_dict(ckpt[key])
    print(f"[diffusion] loaded '{key}' from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model.to(device).eval()


# Data
class CifarSubsetFolder(ImageFolder):

    def find_classes(self, directory):
        classes = sorted(e.name for e in os.scandir(directory) if e.is_dir())
        class_to_idx = {c: int(c.split("_")[1]) for c in classes}
        return classes, class_to_idx


def load_subset(root: str, verbose: bool = True):
    from collections import Counter
    ds = CifarSubsetFolder(root, transform=transforms.Compose([transforms.ToTensor()]))
    if verbose:
        print(f"[data] {root}: {len(ds)} images | class_to_idx = {ds.class_to_idx}")
        print(f"[data] distribution: {dict(sorted(Counter(ds.targets).items()))}")
    return ds

# Three refine modes for the image modality.
#   paper   : noise canvas, t0=1.0, no anchor  
#   repaint : noise canvas, t0=1.0, anchored   -> keep x_bar, regenerate context
#   refine  : local image canvas, t0=0.5, anchored -> default
REFINE_MODES = {
    "paper":   dict(canvas="noise", strength=1.0, forward_diffuse=False, reinject=False),
    "repaint": dict(canvas="noise", strength=1.0, forward_diffuse=False, reinject=True),
    "refine":  dict(canvas="local", strength=0.5, forward_diffuse=True,  reinject=True),
}
DEFAULT_MODE = "refine"

DIFF_STEPS = 500     
DIFF_ETA = 1.0      
GEN_BATCH = 32
N_PER_CLASS = 200    
FEATHER = 1          

# FedCOG 
FEDCOG_START = 50      # --start_round
FEDCOG_N = 256         # --synthesis_batch_size
FEDCOG_STEPS = 500     # --g_steps
FEDCOG_LR = 0.01       # --lr_g
FEDCOG_ADV = 0.1       # --adv (weight of the disagreement loss)
FEDCOG_T = 1.0         # --T / --js_T
FEDCOG_MEAN = (0.4914, 0.4822, 0.4465)  
FEDCOG_STD = (0.2023, 0.1994, 0.2010)
FEDCOG_CHUNK = 256     
                      
FEDCOG_PARALLEL = 1    # one client per GPU; other clients wait for the lock

DEVICE = get_device()
CLIENT_LOG_DIR = os.environ.get("FLGENAI_LOG_DIR", "./client_logs")

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8081
SERVER_ADDRESS = f"{SERVER_IP}:{SERVER_PORT}"


DIFFUSION_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar_diffusion.pth")


def collate_labels_as_long(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.as_tensor([int(y) for y in ys], dtype=torch.long)


class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(self, images: torch.Tensor, labels: List[int]):
        assert len(images) == len(labels)
        self.images = images
        self.labels = [int(v) for v in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]

    @property
    def tensors(self):
        return self.images, torch.tensor(self.labels, dtype=torch.long)


def build_local_pool(dataset, max_per_class: int = 256):
    pool = defaultdict(list)
    for i in range(len(dataset)):
        x, y = dataset[i]
        y = int(y)
        if len(pool[y]) < max_per_class:
            pool[y].append(x)
    return {k: torch.stack(v) for k, v in pool.items()}


def feather_mask(mask: torch.Tensor, radius: int = FEATHER) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.avg_pool2d(F.pad(mask, (radius,) * 4, mode="replicate"), k, stride=1)


def init_and_overwrite(fb_batch: List[Dict[str, Any]], canvas: str = "noise",
                       local_pool=None, device=None):
    device = device or DEVICE
    b = len(fb_batch)
    x0   = torch.randn(b, 3, IMG_SIZE, IMG_SIZE, device=device)          
    mask = torch.zeros(b, 1, IMG_SIZE, IMG_SIZE, device=device)

    if canvas == "local" and local_pool:
        for j, fb in enumerate(fb_batch):
            imgs = local_pool.get(int(fb["label"]))
            if imgs is not None and len(imgs):        
                x0[j] = to_diff(imgs[random.randrange(len(imgs))].to(device))

    for j, fb in enumerate(fb_batch):
        p = int(fb["patch"])
        vals = to_diff(torch.tensor(fb["values"], dtype=torch.float32, device=device))
        for (r, c), v in zip(fb["positions"], vals):                   
            x0[j, :, r * p:(r + 1) * p, c * p:(c + 1) * p] = v
            mask[j, :, r * p:(r + 1) * p, c * p:(c + 1) * p] = 1.0

    return x0, mask, feather_mask(mask) * mask


@torch.no_grad()
def refine(model, x0, classes, steps: int = DIFF_STEPS, eta: float = DIFF_ETA,
           strength: float = 1.0, forward_diffuse: bool = True,
           x_bar=None, mask=None):

    t = torch.linspace(strength, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    x = x0 * alphas[0] + torch.randn_like(x0) * sigmas[0] if forward_diffuse else x0
    ts = x.new_ones([x.shape[0]])

    for i in trange(steps, leave=False):
        with torch.amp.autocast("cuda", enabled=x.is_cuda):
            v = model(x, ts * log_snrs[i], classes).float()

        pred = x * alphas[i] - v * sigmas[i]
        eps  = x * sigmas[i] + v * alphas[i]

        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()
            x = pred * alphas[i + 1] + eps * adjusted_sigma
            if eta:
                x = x + torch.randn_like(x) * ddim_sigma

            if x_bar is not None:      # anchor the Grad-CAM region to x̄
                known = x_bar * alphas[i + 1] + torch.randn_like(x) * sigmas[i + 1]
                x = x * (1 - mask) + known * mask
    return pred


def generate_from_feedback(diffusion, feedback: List[Dict[str, Any]],
                           n_per_class: int = N_PER_CLASS, steps: int = DIFF_STEPS,
                           eta: float = DIFF_ETA, batch_size: int = GEN_BATCH,
                           mode: str = DEFAULT_MODE, local_pool=None,
                           device=None, log_dir=None):
    device = device or DEVICE
    if mode not in REFINE_MODES:
        print(f"[Alg2] mode '{mode}' does not exist, using '{DEFAULT_MODE}' "
              f"(valid: {sorted(REFINE_MODES)})")
        mode = DEFAULT_MODE
    cfg = REFINE_MODES[mode]
    diffusion.to(device).eval()

    by_class = defaultdict(list)
    for fb in feedback:                                   # group F by label y
        by_class[int(fb["label"])].append(fb)

    imgs, lbls = [], []                                  
    for y, items in by_class.items():
        made, cursor = 0, 0
        has_local = bool(local_pool and len(local_pool.get(y, [])))
        canvas = cfg["canvas"] if (cfg["canvas"] == "noise" or has_local) else "noise"
        print(f"[Alg2/{mode}] class {y}: {n_per_class} images | {len(items)} feedback elements "
              f"| canvas={canvas} t0={cfg['strength']}")

        while made < n_per_class:                         
            b = min(batch_size, n_per_class - made)
            batch = [items[(cursor + i) % len(items)] for i in range(b)]  
            cursor += b

            x0, m_hard, m_soft = init_and_overwrite(batch, canvas, local_pool, device)  
            y_t = torch.full((b,), y, dtype=torch.long, device=device)
            gen = refine(diffusion, x0, y_t, steps=steps, eta=eta,                     
                         strength=cfg["strength"], forward_diffuse=cfg["forward_diffuse"],
                         x_bar=x0 if cfg["reinject"] else None,
                         mask=m_soft if cfg["reinject"] else None)

            imgs.append(from_diff(gen).cpu())
            lbls.extend([y] * b)
            made += b

        if log_dir:
            d = os.path.join(log_dir, f"class_{y}")
            os.makedirs(d, exist_ok=True)
            save_image(imgs[-1][:16], os.path.join(d, f"{mode}_preview.png"), nrow=4)

    if not imgs:
        return None
    return SyntheticDataset(torch.cat(imgs, 0), lbls)


def generate_without_feedback(diffusion, classes_to_generate: Dict[int, int],
                              steps: int = DIFF_STEPS, eta: float = DIFF_ETA,
                              batch_size: int = GEN_BATCH, device=None, log_dir=None):

    device = device or DEVICE
    diffusion.to(device).eval()
    imgs, lbls = [], []

    for y, n in classes_to_generate.items():
        made = 0
        print(f"[NoFB] class {y}: {n} images (pure class-conditional)")
        while made < n:
            b = min(batch_size, n - made)
            z = torch.randn(b, 3, IMG_SIZE, IMG_SIZE, device=device)
            y_t = torch.full((b,), int(y), dtype=torch.long, device=device)
            gen = refine(diffusion, z, y_t, steps=steps, eta=eta,
                         strength=1.0, forward_diffuse=False)
            imgs.append(from_diff(gen).cpu())
            lbls.extend([int(y)] * b)
            made += b

        if log_dir:
            d = os.path.join(log_dir, f"class_{y}")
            os.makedirs(d, exist_ok=True)
            save_image(imgs[-1][:16], os.path.join(d, "nofeedback_preview.png"), nrow=4)

    if not imgs:
        return None
    return SyntheticDataset(torch.cat(imgs, 0), lbls)


# FedCOG baseline (Ye et al., "Fake It Till Make It: Federated Learning with
# Consensus-Oriented Generation"

def fedcog_js_adv(s_out, t_out, T: float = FEDCOG_T):

    kl = nn.KLDivLoss(reduction="batchmean")
    P = F.softmax(s_out / T, dim=1)
    Q = F.softmax(t_out / T, dim=1)
    M = 0.5 * (P + Q)
    P, Q, M = P.clamp(0.01, 0.99), Q.clamp(0.01, 0.99), M.clamp(0.01, 0.99)
    js = 0.5 * kl(torch.log(P), M) + 0.5 * kl(torch.log(Q), M)
    return 1.0 - js.clamp(0.0, 1.0)


def fedcog_kd(s_logits, t_logits, T: float = FEDCOG_T):
    return F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1),
                    reduction="batchmean") * (T * T)


def fedcog_label_distribution(class_counts: Dict[int, int]):

    counts = torch.tensor([class_counts.get(c, 0) for c in range(NUM_CLASSES)], dtype=torch.float)
    gap = counts.max() - counts
    return (gap / gap.sum() if gap.sum() > 0 else gap), float(gap.sum())


def fedcog_generate(global_model: nn.Module, local_model: nn.Module, distribution,
                    n: int = FEDCOG_N, steps: int = FEDCOG_STEPS, lr: float = FEDCOG_LR,
                    adv: float = FEDCOG_ADV, T: float = FEDCOG_T, device=None,
                    chunk: int = FEDCOG_CHUNK):

    device = device or DEVICE
    y = torch.cat([torch.full((int(n * float(distribution[c])),), c, dtype=torch.long)
                   for c in range(NUM_CLASSES)]).to(device)
    if len(y) == 0:
        return None
    g = copy.deepcopy(global_model).to(device).eval().requires_grad_(False)
    l = copy.deepcopy(local_model).to(device).eval().requires_grad_(False)

    mean = torch.tensor(FEDCOG_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(FEDCOG_STD, device=device).view(1, 3, 1, 1)
    xs = []
    for yc in y.split(chunk):
        z = torch.randn(len(yc), 3, IMG_SIZE, IMG_SIZE, device=device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr, betas=(0.5, 0.9), eps=1e-8)
        for _ in range(steps):
            x = z * std + mean
            t_out, s_out = g(x), l(x)
            loss = (F.cross_entropy(t_out, yc) + adv * fedcog_js_adv(s_out, t_out, T)) * (len(yc) / len(y))
            opt.zero_grad(); loss.backward(); opt.step()
        xs.append((z * std + mean).detach())
    del g, l
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(xs)


@contextmanager
def fedcog_gpu_lock(device, cid: str, slots: int = FEDCOG_PARALLEL):
    try:
        import fcntl
    except ImportError:
        fcntl = None
    if slots <= 0 or fcntl is None or torch.device(device).type != "cuda":
        yield
        return
    name = str(device).replace(":", "")
    files = [open(os.path.join(tempfile.gettempdir(), f"flgenai_fedcog_{name}_{i}.lock"), "w")
             for i in range(slots)]
    held, t0 = None, time.time()
    print(f"[Client {cid}] FedCOG: waiting for a slot on {device} ...")
    try:
        while held is None:
            for f in files:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = f
                    break
                except BlockingIOError:
                    pass
            else:
                time.sleep(1)
        print(f"[Client {cid}] FedCOG: got slot {files.index(held)}/{slots} on {device} "
              f"after {time.time() - t0:.0f}s")
        yield
    finally:
        if held is not None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()                 
            fcntl.flock(held, fcntl.LOCK_UN)
        for f in files:
            f.close()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, local_dataset, valloader, diffusion_model, cid: str,
                 client_type: str = "train", batch_size: int = 32,
                 local_epochs: int = 2, lr: float = 1e-4, mode: str = DEFAULT_MODE):
        self.local_dataset = local_dataset        
        self.valloader     = valloader
        self.diffusion     = diffusion_model
        self.cid           = cid
        self.client_type   = client_type
        self.batch_size    = batch_size
        self.local_epochs  = local_epochs
        self.lr            = lr
        self.mode          = mode
        self.device        = DEVICE
        self._local_pool   = None      
        self._prev_local   = None      

        self.local_model = resnet18(NUM_CLASSES).to(self.device)
        self.criterion   = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------- Flower
    @property
    def local_pool(self):
        if self._local_pool is None:
            self._local_pool = build_local_pool(self.local_dataset)
            print(f"[Client {self.cid}] canvas pool: "
                  f"{ {k: len(v) for k, v in sorted(self._local_pool.items())} }")
        return self._local_pool

    @property
    def local_class_counts(self) -> Dict[int, int]:
        if not hasattr(self, "_class_counts"):
            ds = self.local_dataset
            if hasattr(ds, "indices") and hasattr(ds.dataset, "targets"):   # random_split Subset
                ys = [int(ds.dataset.targets[i]) for i in ds.indices]
            else:
                ys = [int(ds[i][1]) for i in range(len(ds))]
            self._class_counts = {c: ys.count(c) for c in range(NUM_CLASSES)}
        return self._class_counts

    def least_represented(self, n: int) -> List[int]:
        counts = self.local_class_counts
        return sorted(counts, key=lambda c: (counts[c], c))[:n]

    def get_properties(self, config):
        return {"type": self.client_type}

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.local_model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        sd = {k: torch.tensor(v, device=self.device)
              for k, v in zip(self.local_model.state_dict().keys(), parameters)}
        self.local_model.load_state_dict(sd, strict=True)


    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]):
        if str(config.get("method", "proposed")) == "fedcog":
            return self._fit_fedcog(parameters, config)
        self.set_parameters(parameters)

        d_aug = self.local_dataset                                  
        n_syn = 0

        gen_classes = json.loads(config.get("gen_classes", "{}"))
        budget = json.loads(config.get("gen_budget", "{}"))
        if budget and not gen_classes:
            gen_classes = {c: int(budget["n_per_class"])
                           for c in self.least_represented(int(budget["n_classes"]))}
            print(f"[Client {self.cid}] (no feedback) own least-represented classes "
                  f"{sorted(gen_classes)} (local counts {self.local_class_counts})")
        if gen_classes and "feedback" not in config:
            d_gen = generate_without_feedback(
                self.diffusion, {int(k): int(v) for k, v in gen_classes.items()},
                steps=int(config.get("diff_steps", DIFF_STEPS)),
                device=self.device,
                log_dir=os.path.join(CLIENT_LOG_DIR, self.cid, "generated"))
            if d_gen is not None:
                d_aug = ConcatDataset([self.local_dataset, d_gen])
                n_syn = len(d_gen)
                print(f"[Client {self.cid}] (no feedback) Daug = "
                      f"{len(self.local_dataset)} real + {n_syn} synthetic")

        feedback = json.loads(config.get("feedback", "[]"))
        if feedback:
            top_m = int(config.get("top_m", len(feedback)))
            feedback = sorted(feedback, key=lambda f: f.get("client_prob", 0.0),
                              reverse=True)[:top_m]

            os.makedirs(CLIENT_LOG_DIR, exist_ok=True)
            with open(os.path.join(CLIENT_LOG_DIR,
                                   f"feedback_{self.cid}.json"), "w") as f:
                json.dump([{k: v for k, v in fb.items() if k != "values"} for fb in feedback],
                          f, indent=2)

            classes = sorted({int(fb["label"]) for fb in feedback})
            print(f"[Client {self.cid}] {len(feedback)} feedback elements, classes {classes}")

            mode = str(config.get("mode", self.mode))
            if mode not in REFINE_MODES:
                print(f"[Client {self.cid}] unknown mode '{mode}' -> using '{DEFAULT_MODE}'")
                mode = DEFAULT_MODE
            d_gen = generate_from_feedback(
                self.diffusion, feedback,
                n_per_class=int(config.get("n_per_class", N_PER_CLASS)),
                steps=int(config.get("diff_steps", DIFF_STEPS)),
                mode=mode,
                local_pool=self.local_pool if REFINE_MODES[mode]["canvas"] == "local" else None,
                device=self.device,
                log_dir=os.path.join(CLIENT_LOG_DIR, self.cid, "generated"),
            )
            if d_gen is not None:
                d_aug = ConcatDataset([self.local_dataset, d_gen])    # line 10
                n_syn = len(d_gen)
                print(f"[Client {self.cid}] Daug = {len(self.local_dataset)} real "
                      f"+ {n_syn} synthetic")

        loader = DataLoader(d_aug, batch_size=self.batch_size, shuffle=True,
                            collate_fn=collate_labels_as_long)
        loss, acc = self.train(self.local_model, loader)              # line 11

        return self.get_parameters(), len(d_aug), {
            "client_id": self.cid, "train_loss": loss, "train_acc": acc,
            "num_synthetic": n_syn,
            "gen_mode": self.mode if feedback else ("none" if not n_syn else "no_feedback"),
        }

    # --------------------------------------------------------------- FedCOG
    def _fit_fedcog(self, parameters, config):
        self.set_parameters(parameters)
        if self.diffusion is not None and next(self.diffusion.parameters()).is_cuda:
            self.diffusion.to("cpu")              
        T = float(config.get("fedcog_T", FEDCOG_T))
        rnd, start = int(config.get("server_round", 0)), int(config.get("fedcog_start", FEDCOG_START))
        generating = rnd >= start and self._prev_local is not None  

        with fedcog_gpu_lock(self.device, self.cid) if generating else nullcontext():
            teacher = copy.deepcopy(self.local_model).eval()      
            x_gen, real_w, syn_w = None, 1.0, 0.0
            if generating:
                dist, num_syn = fedcog_label_distribution(self.local_class_counts)
                x_gen = fedcog_generate(
                    teacher, self._prev_local, dist,
                    n=int(config.get("fedcog_n", FEDCOG_N)),
                    steps=int(config.get("fedcog_steps", FEDCOG_STEPS)),
                    lr=float(config.get("fedcog_lr", FEDCOG_LR)),
                    adv=float(config.get("fedcog_adv", FEDCOG_ADV)), T=T, device=self.device,
                    chunk=int(config.get("fedcog_chunk", FEDCOG_CHUNK)))
                if x_gen is not None:
                    num_real = len(self.local_dataset)
                    real_w, syn_w = num_real / (num_real + num_syn), num_syn / (num_real + num_syn)
                    d = os.path.join(CLIENT_LOG_DIR, self.cid, "generated")
                    os.makedirs(d, exist_ok=True)
                    save_image(x_gen[:16].cpu(), os.path.join(d, "fedcog_preview.png"), nrow=4,
                               normalize=True)
                    print(f"[Client {self.cid}] FedCOG: {len(x_gen)} consensus samples, "
                          f"label dist {[round(float(v), 2) for v in dist]}, "
                          f"real_lr {real_w:.3f} syn_lr {syn_w:.3f}")

            loader = DataLoader(self.local_dataset, batch_size=self.batch_size, shuffle=True,
                                collate_fn=collate_labels_as_long)
            loss, acc = self.train_fedcog(self.local_model, loader, teacher, x_gen, real_w, syn_w, T)
            num_synthetic = 0 if x_gen is None else len(x_gen)
            self._prev_local = copy.deepcopy(self.local_model).cpu().eval()
            del teacher, x_gen, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.get_parameters(), len(self.local_dataset), {
            "client_id": self.cid, "train_loss": loss, "train_acc": acc,
            "num_synthetic": num_synthetic, "gen_mode": "fedcog",
        }

    def train_fedcog(self, model, loader, teacher, x_gen, real_w: float, syn_w: float,
                     T: float = FEDCOG_T):

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        total_loss = correct = seen = 0

        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out  = model(x)
                ce   = self.criterion(out, y)
                loss = ce
                if x_gen is not None:
                    idx = torch.randint(0, len(x_gen), (len(x),), device=x_gen.device)
                    xg = x_gen[idx]
                    with torch.no_grad():
                        t_logits = teacher(xg)
                    loss = real_w * ce + syn_w * fedcog_kd(model(xg), t_logits, T)

                opt.zero_grad(); loss.backward(); opt.step()

                total_loss += ce.item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                seen       += y.size(0)

        return total_loss / max(seen, 1), correct / max(seen, 1)

    def train(self, model, loader):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        total_loss = correct = seen = 0

        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out  = model(x)
                loss = self.criterion(out, y)

                opt.zero_grad(); loss.backward(); opt.step()

                total_loss += loss.item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                seen       += y.size(0)

        return total_loss / max(seen, 1), correct / max(seen, 1)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.local_model.eval()
        loss = correct = seen = 0

        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(self.device), y.to(self.device)
                out  = self.local_model(x)
                loss += self.criterion(out, y).item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                seen += y.size(0)

        return loss / max(seen, 1), seen, {"accuracy": correct / max(seen, 1)}


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FLGenAI client")
    p.add_argument("--data", required=True, help="client's non-IID subset directory")
    p.add_argument("--ckpt", default=DIFFUSION_CKPT,
                   help="diffusion checkpoint (default: cifar_diffusion.pth next to client.py)")
    p.add_argument("--cid", required=True, help="client identifier")
    p.add_argument("--mode", default=DEFAULT_MODE, choices=sorted(REFINE_MODES),
                   help="image generation mode; the server may override it via config")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-dir", default=CLIENT_LOG_DIR)
    return p.parse_args(argv)


def main(argv=None):
    global DEVICE, CLIENT_LOG_DIR

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(argv)
    set_seed(args.seed)
    DEVICE = get_device(args.device)
    CLIENT_LOG_DIR = args.log_dir
    os.makedirs(CLIENT_LOG_DIR, exist_ok=True)
    print(f"[client {args.cid}] device={DEVICE} mode={args.mode}")

    dataset = load_subset(args.data)
    n_val = max(1, int(args.val_split * len(dataset)))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    valloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_labels_as_long)

    client = FlowerClient(
        local_dataset=train_ds,
        valloader=valloader,
        diffusion_model=load_diffusion(args.ckpt, DEVICE),
        cid=args.cid,
        client_type="train",
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        mode=args.mode,
    )

    print(f"[client {args.cid}] connecting to {SERVER_ADDRESS} ...")
    # flwr>=1.7: start_numpy_client is deprecated -> start_client + to_client()
    fl.client.start_client(server_address=SERVER_ADDRESS, client=client.to_client())


if __name__ == "__main__":
    main()
