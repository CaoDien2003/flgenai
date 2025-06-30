import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr.server.strategy import FedAvg
import numpy as np
import json
import os
from collections import defaultdict
import logging
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.datasets as datasets

class Config:
    # Đường dẫn và thư mục
    SERVER_LOG_DIR = "~/server_logs"
    CHECKPOINT_DIR = "~/checkpoint2"
    TEST_DATA_DIR = "~/subset1"
    RESUME_CHECKPOINT = "~/checkpoint2/global_round_n.pth"
    
    NUM_CLASSES = 10
    INPUT_SIZE = (32, 32)
    BATCH_SIZE = 32
    NUM_ROUNDS = 50
    MIN_FIT_CLIENTS = 1
    MIN_AVAILABLE_CLIENTS = 1
    SERVER_ADDRESS = "0.0.0.0:8081"
    
    TOP_CORRECT = 50
    TOP_WRONG = 50
    CLASS_GROUP_SIZE = 10

    LOG_LEVEL = logging.INFO

config = Config()

os.makedirs(config.SERVER_LOG_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.SERVER_LOG_DIR, "server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FlowerServer")

class CifarResNet18(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

features = None
grads = None

def save_features_hook(module, input, output):
    global features
    features = output

def save_grads_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0]

def compute_gradcam():
    global features, grads
    if grads is None or features is None:
        return torch.zeros((1, 1), device=device)
    
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    heatmap = torch.zeros(features.shape[2:], device=features.device)
    for i in range(features.shape[1]):
        heatmap += pooled_grads[i] * features[0, i, :, :]
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max() + 1e-8
    return heatmap

transform = transforms.Compose([
    transforms.Resize(config.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    testset = datasets.ImageFolder(root=config.TEST_DATA_DIR, transform=transform)
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=False)
    logger.info(f"Loaded test dataset with {len(testset)} samples")
except Exception as e:
    logger.error(f"Error loading test dataset: {e}")
    raise

def get_class_group(server_round: int, num_classes=config.NUM_CLASSES, group_size=config.CLASS_GROUP_SIZE):
    num_groups = num_classes // group_size
    group_idx = (server_round - 1) % num_groups
    start = group_idx * group_size
    return list(range(start, min(start + group_size, num_classes)))

def load_initial_parameters():
    try:
        if os.path.exists(config.RESUME_CHECKPOINT):
            model = CifarResNet18()
            checkpoint = torch.load(config.RESUME_CHECKPOINT, map_location="cpu")
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded initial parameters from {config.RESUME_CHECKPOINT}")
            return fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for val in model.state_dict().values()]
            )
    except Exception as e:
        logger.error(f"Error loading initial parameters: {e}")
    
    # Fallback to new model
    model = CifarResNet18()
    logger.info("Using new model parameters")
    return fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )

# ======================== FEEDBACK ========================
feedback_storage = {}

def generate_feedback_for_client(client_id, model, server_round, dataset, test_acc=None):
    logger.info(f"Generating feedback for Client {client_id}, Test Acc: {test_acc:.2f}%")
    
    try:
        model.to(device)
        model.eval()
        
        # Hook layer
        last_conv_layer = model.backbone.layer4[-1].conv2
        handle_fwd = last_conv_layer.register_forward_hook(save_features_hook)
        handle_bwd = last_conv_layer.register_full_backward_hook(save_grads_hook)
        
        class_results = defaultdict(lambda: {"correct": [], "wrong": []})
        selected_classes = get_class_group(server_round)
        logger.info(f"Selected classes for round {server_round}: {selected_classes}")

        for img_tensor, true_label in tqdm(dataset, desc=f"Grad-CAM Client {client_id}"):
            if true_label not in selected_classes:
                continue
                
            img_tensor = img_tensor.to(device)
            input_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
            
            #predict
            with torch.no_grad():
                output = model(input_tensor)
                pred_label = output.argmax().item()
                confidence = torch.softmax(output, dim=1)[0, pred_label].item()
            
            # gradient
            model.zero_grad()
            output = model(input_tensor)
            output[0, true_label].backward()
            
            # heatmap
            heatmap = compute_gradcam().detach().cpu().numpy()
            heatmap = np.nan_to_num(heatmap)
            
            # important region
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            box_size = 10
            top = max(y - box_size//2, 0)
            left = max(x - box_size//2, 0)
            bottom = min(y + box_size//2, heatmap.shape[0])
            right = min(x + box_size//2, heatmap.shape[1])
            position = [int(top), int(left), int(bottom), int(right)]
            
            key = "correct" if pred_label == true_label else "wrong"
            result = {"confidence": confidence, "heatmap": heatmap.tolist(), "position": position}
            if key == "wrong":
                result["predicted_label"] = pred_label
            
            class_results[true_label][key].append(result)
        
        # Tạo feedback
        feedback = []
        for class_idx, results in class_results.items():
            correct_samples = sorted(results["correct"], key=lambda x: -x["confidence"])[:config.TOP_CORRECT]
            wrong_samples = sorted(results["wrong"], key=lambda x: -x["confidence"])[:config.TOP_WRONG]
            
            if correct_samples:
                feedback.append({
                    "type": "correct_ref",
                    "target_class": class_idx,
                    "refs": [{"heatmap": s["heatmap"], "position": s["position"]} for s in correct_samples]
                })
            
            if wrong_samples:
                feedback.append({
                    "type": "wrong_samples",
                    "true_label": class_idx,
                    "samples": [{"predicted_label": s["predicted_label"], 
                                "heatmap": s["heatmap"], 
                                "position": s["position"]} for s in wrong_samples]
                })
        
        # Lưu feedback
        feedback_storage[client_id] = feedback
        logger.info(f"Generated {len(feedback)} feedback items for client {client_id}")
        
        return feedback
    except Exception as e:
        logger.error(f"Error generating feedback for client {client_id}: {e}")
        return []
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

# ======================== Model Eval ========================
def evaluate_global_model(server_round, parameters, config):
    try:
        model = CifarResNet18().to(device)
        state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() * data.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(testset)
        
        logger.info(f"Global model round {server_round} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")
        return avg_loss, {"accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error evaluating global model: {e}")
        return float("inf"), {"accuracy": 0.0}

# ======================== STRATEGY ========================
class FeedbackFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_id_map = {}
        self.accuracy_history = []

    def evaluate(self, server_round, parameters):
        loss, metrics = evaluate_global_model(server_round, parameters, config)
        acc = metrics.get("accuracy", 0.0)
        self.accuracy_history.append(acc)
        return loss, metrics

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins = []
        clients = list(client_manager.all().values())
        
        for client in clients:
            client_id = self.client_id_map.get(client.cid, f"client_{len(self.client_id_map)}")
            self.client_id_map[client.cid] = client_id
            
            feedback = feedback_storage.pop(client_id, [])
            feedback_json = json.dumps(feedback) if feedback else "[]"
            
            fit_ins.append((
                client,
                fl.common.FitIns(
                    parameters,
                    {"client_id": client_id, "feedback": feedback_json}
                )
            ))
            logger.info(f"Sending feedback to {client_id} ({len(feedback)} items)")
        
        return fit_ins

    def aggregate_fit(self, server_round, results, failures):
        if len(self.accuracy_history) > 1 and self.accuracy_history[-1] < self.accuracy_history[-2]:
            logger.info("Accuracy dropped from previous round - triggering feedback immediately")
            for client_proxy, fit_res in results:
                try:
                    client_id = fit_res.metrics.get("client_id", f"client_{client_proxy.cid}")
                    weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    model = CifarResNet18().to(device)
                    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
                    model.load_state_dict(state_dict)

                    correct, total = 0, 0
                    with torch.no_grad():
                        for x, y in testloader:
                            x, y = x.to(device), y.to(device)
                            outputs = model(x)
                            preds = outputs.argmax(dim=1)
                            correct += (preds == y).sum().item()
                            total += y.size(0)
                    test_acc = correct / total

                    generate_feedback_for_client(client_id, model, server_round, testset, test_acc)
                except Exception as e:
                    logger.error(f"Error generating feedback for {client_id}: {e}")

        if server_round % 5 == 0 or server_round == config.NUM_ROUNDS:
            try:
                weights = fl.common.parameters_to_ndarrays(results[0][1].parameters)
                state_dict = {k: torch.tensor(v) for k, v in zip(CifarResNet18().state_dict().keys(), weights)}
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"global_round_{server_round}.pth")
                torch.save(state_dict, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")

        return super().aggregate_fit(server_round, results, failures)


# ======================== SERVER ========================
if __name__ == "__main__":
    strategy = FeedbackFedAvg(
        fraction_fit=1.0,
        min_fit_clients=config.MIN_FIT_CLIENTS,
        min_available_clients=config.MIN_AVAILABLE_CLIENTS,
        evaluate_fn=evaluate_global_model,
        initial_parameters=load_initial_parameters()
    )

    fl.server.start_server(
        server_address=config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy
    )
