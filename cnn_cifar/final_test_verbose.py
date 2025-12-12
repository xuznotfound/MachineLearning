# -*- coding: utf-8 -*-
"""
PyTorch CIFAR-10 ResNet-like CNN with residual blocks
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for saving figures
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理：与常见CIFAR-10设置一致（裁剪+翻转+均值方差归一化）
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def format_seconds(seconds: float) -> str:
    """Human-friendly time formatter for logs."""
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:05.2f}s" if seconds >= 60 else f"{s:0.2f}s"


# 残差块定义（2个3x3卷积，必要时用1x1卷积对齐维度/步幅）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果通道或步幅不匹配，用1x1卷积做shortcut以保持形状一致
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# ResNet-like 主干：4个stage，通道数64/128/256/512，对应stride=1/2/2/2
class ResNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每若干batch输出一次训练进度，避免刷屏
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            running_acc = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  [Epoch {epoch:03d}][Batch {batch_idx+1:04d}/{len(train_loader):04d}] "
                  f"Loss {avg_loss:.4f} | Acc {running_acc:.2f}%")

    epoch_time = time.time() - epoch_start
    train_acc = 100.0 * correct / total
    train_loss = total_loss / len(train_loader)

    # 统计吞吐：样本/秒
    samples_per_sec = total / epoch_time if epoch_time > 0 else 0.0
    return train_acc, train_loss, epoch_time, samples_per_sec


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100.0 * correct / total
    test_loss = total_loss / len(test_loader)
    return test_acc, test_loss


if __name__ == '__main__':
    # 1) 设备选择与信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 2) 数据集与DataLoader（多进程+pin_memory加速主机->GPU传输）
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 3) 模型、损失、优化器、学习率调度
    model = ResNetCIFAR10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    num_epochs = 200
    best_test_acc = 0.0

    # Track metrics per epoch for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': [],
        'epoch_time': [],
        'throughput': []
    }

    print("开始训练...")
    for epoch in range(1, num_epochs + 1):
        # 当前学习率（Adam可用第一个param_group的lr）
        current_lr = optimizer.param_groups[0]['lr']

        train_acc, train_loss, epoch_time, samples_per_sec = train(model, train_loader, criterion, optimizer, epoch, device)
        test_acc, test_loss = test(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        history['throughput'].append(samples_per_sec)

        print(f"Epoch {epoch:03d} | LR {current_lr:.5f} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
              f"Test Loss {test_loss:.4f} Acc {test_acc:.2f}% | "
              f"Epoch Time {format_seconds(epoch_time)} | Throughput {samples_per_sec:0.1f} samples/s")

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_acc': best_test_acc
            }, 'resnet_cifar10_gpu_best_verbose.pth')
            print(f"  >>> 保存最优模型，当前最佳测试准确率: {best_test_acc:.2f}%")

    print(f"训练结束，最佳测试准确率: {best_test_acc:.2f}%")

    # Plot curves with English labels
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['test_loss'], label='Test Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(epochs, history['train_acc'], label='Train Accuracy')
    axes[1].plot(epochs, history['test_acc'], label='Test Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    curves_path = Path('training_curves.png')
    fig.savefig(curves_path, dpi=150)
    print(f"Saved training curves to: {curves_path.resolve()}")

    # Save final model checkpoint
    final_path = Path('resnet_cifar10_gpu_final_verbose.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': best_test_acc,
        'history': history
    }, final_path)
    print(f"Saved final model to: {final_path.resolve()}")
