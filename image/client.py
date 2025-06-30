import os
import json
import torch
import flwr as fl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from collections import defaultdict
from tqdm import trange
import math
import random
from typing import Dict, List, Union
from torchvision.utils import save_image

class ClientConfig:
    CLIENT_LOG_DIR = "~/client_logs"
    CHECKPOINT_DIR = "~/client_model_checkpoints"
    GENERATED_DIR = "~/generated"
    
    BATCH_SIZE = 32
    INPUT_SIZE = (32, 32)
    LEARNING_RATE = 0.001
    MU = 0.01  
    EPOCHS = 1
    
    SYNTH_STEPS = 250
    SYNTH_ETA = 1.0
    NUM_PER_CLASS = 250
    
    NUM_CLASSES = 10
    
    CHECKPOINT_INTERVAL = 5 #save checkpoint


config = ClientConfig()

os.makedirs(config.CLIENT_LOG_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.GENERATED_DIR, exist_ok=True)

class CifarResNet18(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ======================== DIFFUSION ========================
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
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
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

def sample_ddpm(model, x, steps, eta, classes):
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t).to(x.device)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    ts = x.new_ones([x.shape[0]])

    for i in trange(steps, desc="Sampling"):
        with torch.cuda.amp.autocast():
            v = model(x, ts * log_snrs[i], classes).float()
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]
        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                          (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()
            x = pred * alphas[i + 1] + eps * adjusted_sigma
            if eta:
                x += torch.randn_like(x) * ddim_sigma
    return pred

def synthesize_from_feedback(feedback_list, diffusion_model, device):
    diffusion_model.eval()
    diffusion_model.to(device)
    
    output_dir = os.path.join(config.GENERATED_DIR, f"client_{device.cid}")
    os.makedirs(output_dir, exist_ok=True)
    
    synthetic_dataset = []
    ref_dict = defaultdict(list)

    for item in feedback_list:
        if item["type"] == "correct_ref":
            cls = item["target_class"]
            ref_dict[cls].extend(item["refs"])

    for cls in ref_dict:
        refs = ref_dict[cls]
        count = 0
        
        while count < config.NUM_PER_CLASS:
            ref = random.choice(refs)
            heatmap_np = np.array(ref["heatmap"])
            y1, x1, y2, x2 = ref["position"]

            x_start = torch.randn(3, 32, 32).to(device)
            
            heatmap_tensor = torch.tensor(heatmap_np).float()
            heatmap_tensor = (heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-5)
            heatmap_tensor = F.interpolate(heatmap_tensor.unsqueeze(0).unsqueeze(0), size=(32, 32), mode="bilinear")
            heatmap_mask = heatmap_tensor.squeeze(0).repeat(3, 1, 1).to(device)

            x_start = x_start * (1 - heatmap_mask) + 0.5 * heatmap_mask

            gen_img = sample_ddpm(
                model=diffusion_model,
                x=x_start.unsqueeze(0),
                steps=config.SYNTH_STEPS,
                eta=config.SYNTH_ETA,
                classes=torch.tensor([cls], device=device)
            ).squeeze(0)
            save_path = os.path.join(output_dir, f"class_{cls}_gen_{count}.png")
            save_image((gen_img + 1) / 2, save_path)
            synthetic_dataset.append((gen_img.detach().cpu(), cls))
            count += 1

    return synthetic_dataset

# ======================== FLOWER CLIENT ========================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, diffusion_model, cid, client_type, resume_checkpoint_path=None):
        self.trainloader = trainloader
        self.diffusion_model = diffusion_model
        self.cid = cid
        self.client_type = client_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.round_counter = 0
        self.local_model = CifarResNet18().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.global_weights = None
        
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            try:
                checkpoint = torch.load(resume_checkpoint_path, map_location=self.device)
                self.local_model.load_state_dict(checkpoint)
                print(f"[Client {self.cid}] Loaded checkpoint from {resume_checkpoint_path}")
            except Exception as e:
                print(f"[Client {self.cid}] Error loading checkpoint: {e}")

    def get_properties(self, config):
        return {"type": self.client_type}

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.local_model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = dict(zip(self.local_model.state_dict().keys(), parameters))
        for k, v in state_dict.items():
            state_dict[k] = torch.tensor(v, device=self.device)
        self.local_model.load_state_dict(state_dict, strict=True)
        self.global_weights = {k: v.detach().clone() for k, v in self.local_model.state_dict().items()}

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Union[str, float, int]]):
        print(f"\n[Client {self.cid}] Starting training round {self.round_counter + 1}")
        
        self.set_parameters(parameters)       
        self.process_feedback_and_augment(config)
        loss_avg, num_examples, metrics = self.train()
        self.round_counter += 1
        if self.round_counter % config.CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint()
        
        return self.get_parameters(), num_examples, metrics

    def process_feedback_and_augment(self, config):
        feedback_raw = config.get("feedback", "[]")
        feedback_list = json.loads(feedback_raw)

        if not feedback_list:
            print(f"[Client {self.cid}] No feedback received.")
            return
        log_path = os.path.join(config.CLIENT_LOG_DIR, f"feedback_received_client_{self.cid}.json")
        with open(log_path, "w") as f:
            json.dump(feedback_list, f, indent=2)
        synthetic_data = synthesize_from_feedback(
            feedback_list=feedback_list,
            diffusion_model=self.diffusion_model,
            device=self.device
        )

        if synthetic_data:
            print(f"[Client {self.cid}] Generated {len(synthetic_data)} synthetic samples")
            self.merge_synthetic_with_train(synthetic_data)

    def merge_synthetic_with_train(self, synthetic_data):
        if not synthetic_data:
            return
            
        imgs_tensor = torch.stack([x[0] for x in synthetic_data])
        labels_tensor = torch.tensor([x[1] for x in synthetic_data])
        synthetic_dataset = TensorDataset(imgs_tensor, labels_tensor)
        merged_dataset = ConcatDataset([self.trainloader.dataset, synthetic_dataset])
        self.trainloader = DataLoader(
            merged_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"[Client {self.cid}] Merged {len(synthetic_dataset)} synthetic samples")
        print(f"[Client {self.cid}] Total samples after merge: {len(merged_dataset)}")

    def train(self):
        self.local_model.train()
        optimizer = torch.optim.Adam(
            self.local_model.parameters(), 
            lr=config.LEARNING_RATE
        )
        prox_reg = 0.0
        for name, param in self.local_model.named_parameters():
            prox_reg += ((param - self.global_weights[name]) ** 2).sum()
        prox_reg *= (config.MU / 2)
        for epoch in range(config.EPOCHS):
            total_loss, correct, total = 0.0, 0, 0
            
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels) + prox_reg
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / total
        
        print(f"[Client {self.cid}] Training complete | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
        
        return avg_loss, total, {
            "accuracy": accuracy, 
            "loss": avg_loss, 
            "client_id": self.cid
        }

    def save_checkpoint(self):
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, 
            f"client_{self.cid}_round_{self.round_counter}.pth"
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.local_model.state_dict(), checkpoint_path)
        print(f"[Client {self.cid}] 💾 Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    CLIENT_ID = "client_1"
    DATA_DIR = "/content/subset3/content/drive/MyDrive/INDEStudy/subset/subset3"
    # RESUME_PATH = "/content/drive/MyDrive/INDEStudy/Clients/client_model_checkpoints/client_client_{client_id}_round_n.pth"
    DIFFUSION_MODEL_PATH = "/content/drive/MyDrive/INDEStudy/cifar_diffusion.pth"
    SERVER_ADDRESS = "128.214.252.95:8081"
    
    transform = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=DATA_DIR, transform=transform)
    trainloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    diffusion_model = Diffusion()
    checkpoint = torch.load(DIFFUSION_MODEL_PATH, map_location="cpu")
    diffusion_model.load_state_dict(checkpoint["model_ema"])
    try:
        checkpoint = torch.load(DIFFUSION_MODEL_PATH, map_location="cpu")
        diffusion_model.load_state_dict(checkpoint["model_ema"])
        print("Diffusion model loaded successfully")
    except Exception as e:
        print(f" Error loading diffusion model: {e}")
        raise
    
    client = FlowerClient(
        trainloader=trainloader,
        diffusion_model=diffusion_model,
        cid=CLIENT_ID,
        client_type="train",
        # resume_checkpoint_path=RESUME_PATH
    )

    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)