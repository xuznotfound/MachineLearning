"""
Assignment 3: Fine-tuning pretrained AlexNet, VGG16, and ResNet18 on CIFAR-10.
Two settings per model:
1) Fine-tune the entire network.
2) Freeze the backbone and only fine-tune the classification head.

Pretrained weights are expected in ../datasets-readonly as provided by the course.
This script is GPU-ready (tested on a 4090) and uses mixed precision when CUDA is available.
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------- utils -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:05.2f}s" if seconds >= 60 else f"{s:0.2f}s"


# ---------------------------- data ------------------------------
def build_dataloaders(data_root: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    test_tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


# ---------------------------- models ----------------------------
MODEL_FILENAMES = {
    "alexnet": "alexnet_pretrained_pytorch.pth",
    "vgg16": "vgg16_pretrained_pytorch.pth",
    "resnet18": "resnet18_pretrained_pytorch.pth",
}


def replace_classifier(model_name: str, model: nn.Module, num_classes: int) -> nn.Module:
    if model_name == "alexnet":
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "vgg16":
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "resnet18":
        model.fc = nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def load_pretrained_backbone(model_name: str, weights_root: Path, num_classes: int, device: torch.device) -> nn.Module:
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        head_keys = ["classifier.6.weight", "classifier.6.bias"]
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        head_keys = ["classifier.6.weight", "classifier.6.bias"]
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        head_keys = ["fc.weight", "fc.bias"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    weights_path = weights_root / MODEL_FILENAMES[model_name]
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        for k in head_keys:
            state_dict.pop(k, None)  # drop head to avoid size mismatch
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading {model_name}: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading {model_name}: {unexpected}")
        print(f"Loaded pretrained weights (backbone only) for {model_name} from {weights_path}")
    else:
        print(f"[WARN] Pretrained weights not found at {weights_path}, using random init")

    model = replace_classifier(model_name, model, num_classes)
    return model.to(device)


def freeze_backbone(model_name: str, model: nn.Module) -> List[nn.Parameter]:
    if model_name in {"alexnet", "vgg16"}:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier.6")
    elif model_name == "resnet18":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    set_batchnorm_eval(model)
    return [p for p in model.parameters() if p.requires_grad]


def unfreeze_all(model: nn.Module) -> List[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = True
    return list(model.parameters())


def set_batchnorm_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


# ---------------------------- training --------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, scaler: GradScaler) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


# ---------------------------- experiments -----------------------
def run_variant(model_name: str, freeze: bool, train_loader: DataLoader, test_loader: DataLoader, cfg: argparse.Namespace, device: torch.device, save_dir: Path) -> Dict:
    mode = "frozen_head" if freeze else "full_finetune"
    model = load_pretrained_backbone(model_name, cfg.weights_root, num_classes=10, device=device)
    params = freeze_backbone(model_name, model) if freeze else unfreeze_all(model)

    lr = cfg.frozen_lr if freeze else cfg.lr
    optimizer = optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = -1.0
    best_path = save_dir / f"{model_name}_{mode}_best.pth"
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model": model_name,
                "mode": mode,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "train_acc": train_acc,
            }, best_path)

        print(f"[{model_name}][{mode}] Epoch {epoch:03d}/{cfg.epochs} | Train {train_loss:.4f}/{train_acc:.2f}% | Test {test_loss:.4f}/{test_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.2e}")

    elapsed = time.time() - start
    print(f"[{model_name}][{mode}] Best test acc: {best_acc:.2f}% | Time: {format_time(elapsed)} | Saved: {best_path}")

    return {
        "mode": mode,
        "best_test_acc": best_acc,
        "history": history,
        "elapsed_sec": elapsed,
        "checkpoint": str(best_path),
    }


def run_all(cfg: argparse.Namespace) -> Dict[str, Dict[str, Dict]]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = build_dataloaders(cfg.data_root, cfg.batch_size, cfg.num_workers)
    results: Dict[str, Dict[str, Dict]] = {}

    for model_name in cfg.models:
        print("=" * 80)
        print(f"Running experiments for {model_name}")
        full = run_variant(model_name, freeze=False, train_loader=train_loader, test_loader=test_loader, cfg=cfg, device=device, save_dir=cfg.save_dir)
        frozen = run_variant(model_name, freeze=True, train_loader=train_loader, test_loader=test_loader, cfg=cfg, device=device, save_dir=cfg.save_dir)
        results[model_name] = {"full_finetune": full, "frozen_head": frozen}

    summary_path = cfg.save_dir / "assignment3_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_path}")
    return results


# ---------------------------- cli --------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 3 finetuning on CIFAR-10")
    parser.add_argument("--models", nargs="+", default=["alexnet", "vgg16", "resnet18"], help="Models to run")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per setting")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for full finetune")
    parser.add_argument("--frozen-lr", type=float, default=1e-3, help="LR when only head is trained")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data", help="CIFAR10 root")
    parser.add_argument("--weights-root", type=Path, default=Path(__file__).parent.parent / "datasets-readonly", help="Pretrained weights root")
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).parent / "outputs", help="Directory to save checkpoints and logs")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    run_all(cfg)
