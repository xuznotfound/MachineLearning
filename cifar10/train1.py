# 1. 加载数据集
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os

# 设置随机种子保证可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 改为True以提高性能

set_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # 设置GPU内存使用策略
    torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存

# 2. 数据增强和预处理
# 更丰富的数据增强策略
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # 注意：RandomErasing在FashionMNIST上可能效果不佳，因为图像简单
    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

# 测试集只需要基础转换
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_data = datasets.FashionMNIST(root='/tmp', download=True, train=True, transform=train_transform)
test_data = datasets.FashionMNIST(root='/tmp', download=True, train=False, transform=test_transform)

# 创建验证集 (10%的训练数据)
val_size = int(0.1 * len(train_data))
train_size = len(train_data) - val_size
train_data, val_data = random_split(train_data, [train_size, val_size])

print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
print(f"测试集大小: {len(test_data)}")

# 3. 创建DataLoader
BATCH_SIZE = 256  # 4090可以处理更大的批次

# 获取CPU核心数
num_workers = min(8, os.cpu_count())

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

# 4. 改进的模型架构
class ImprovedFashionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImprovedFashionCNN, self).__init__()
        
        # 第一卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 第二卷积块
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 第三卷积块
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 10)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # 第二卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # 第三卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

# 5. 训练和测试函数（改进版）
def train_epoch(model, data_loader, loss_fn, optimizer, device, scaler, scheduler=None):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(data_loader, desc='训练中', leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零
        
        # 使用混合精度训练
        with autocast():
            output = model(data)
            loss = loss_fn(output, target)
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率（如果是步进调度器）
        if scheduler is not None and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        if batch_idx % 10 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = train_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, data_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='验证中', leave=False)
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = loss_fn(output, target)
            
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if len(pbar) > 10 and (len(pbar) % 10 == 0):
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    
    avg_loss = val_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_targets

def test_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='测试中', leave=False)
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return all_preds, all_targets

# 6. 模型训练循环
def train_model(model, train_loader, val_loader, test_loader, device, epochs=50):
    # 损失函数（带标签平滑）
    try:
        # PyTorch 1.10+ 支持标签平滑
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    except:
        # 旧版本不支持标签平滑
        loss_fn = nn.CrossEntropyLoss()
    
    # 优化器（AdamW + 权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器 - 移除verbose参数
    try:
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    except TypeError as e:
        # 如果还有参数问题，使用最简单的配置
        print(f"警告: {e}, 使用简化的调度器配置")
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 混合精度训练的梯度缩放器
    scaler = GradScaler()
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    # 早停机制
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    print("开始训练...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, scheduler_cosine
        )
        
        # 验证阶段
        val_loss, val_acc, val_preds, val_targets = validate(
            model, val_loader, loss_fn, device
        )
        
        # 更新学习率（基于plateau）
        try:
            scheduler_plateau.step(val_loss)
        except:
            # 如果调度器调用失败，手动调整
            if epoch > 0 and val_loss > history['val_loss'][-1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n⚠ 早停触发！最佳验证准确率: {best_val_acc:.2f}%")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 最终测试
    print("\n" + "="*50)
    print("最终测试")
    test_preds, test_targets = test_model(model, test_loader, device)
    test_acc = accuracy_score(test_targets, test_preds) * 100
    
    print(f"测试准确率: {test_acc:.2f}%")
    print("分类报告:")
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))
    
    return model, history, test_preds, test_targets, test_acc

# 7. 可视化函数
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练和验证损失
    axes[0, 0].plot(history['train_loss'], label='训练损失', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练和验证准确率
    axes[0, 1].plot(history['train_acc'], label='训练准确率', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='验证准确率', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].set_title('训练和验证准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率变化
    axes[1, 0].plot(history['lr'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('学习率')
    axes[1, 0].set_title('学习率变化')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 训练vs验证准确率散点图
    axes[1, 1].scatter(history['train_acc'], history['val_acc'], alpha=0.6, s=50)
    min_val = min(min(history['train_acc']), min(history['val_acc'])) - 1
    max_val = max(max(history['train_acc']), max(history['val_acc'])) + 1
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5)
    axes[1, 1].set_xlabel('训练准确率 (%)')
    axes[1, 1].set_ylabel('验证准确率 (%)')
    axes[1, 1].set_title('训练vs验证准确率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数'})
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_misclassified_examples(model, data_loader, device, class_names, num_examples=10):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            _, pred = output.max(1)
            
            mask = pred != target
            misclassified_data = data[mask]
            misclassified_pred = pred[mask]
            misclassified_target = target[mask]
            
            for i in range(len(misclassified_data)):
                if len(misclassified) < num_examples:
                    misclassified.append((
                        misclassified_data[i].cpu(),
                        misclassified_pred[i].cpu().item(),
                        misclassified_target[i].cpu().item()
                    ))
                else:
                    break
            
            if len(misclassified) >= num_examples:
                break
    
    # 绘制错误分类的样本
    if misclassified:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for idx, (img, pred_label, true_label) in enumerate(misclassified[:num_examples]):
            axes[idx].imshow(img.squeeze(), cmap='gray')
            axes[idx].set_title(f'预测: {class_names[pred_label]}\n真实: {class_names[true_label]}', 
                               fontsize=10, color='red' if pred_label != true_label else 'green')
            axes[idx].axis('off')
        
        plt.suptitle('错误分类示例', fontsize=16)
        plt.tight_layout()
        plt.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("没有找到错误分类的样本！")

# 8. 模型集成（可选）
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 平均所有模型的输出
        avg_output = torch.stack(outputs).mean(0)
        return avg_output

# 9. 主程序
def main():
    # 类别名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 创建模型
    print("创建模型...")
    model = ImprovedFashionCNN(dropout_rate=0.3).to(device)
    
    # 打印模型摘要
    try:
        from torchinfo import summary
        summary(model, (BATCH_SIZE, 1, 28, 28), 
                col_names=["input_size", "output_size", "num_params", "trainable"],
                verbose=0)
    except:
        print("无法导入torchinfo，跳过模型摘要")
        # 手动打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    # 训练模型
    model, history, test_preds, test_targets, test_acc = train_model(
        model, train_dataloader, val_dataloader, test_dataloader, device, epochs=50
    )
    
    # 可视化结果
    plot_training_history(history)
    plot_confusion_matrix(test_targets, test_preds, class_names)
    plot_misclassified_examples(model, test_dataloader, device, class_names)
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'class_names': class_names,
        'history': history
    }, 'final_model.pth')
    print(f"\n模型已保存，测试准确率: {test_acc:.2f}%")
    
    # 返回结果
    return model, history, test_acc

# 10. 运行主程序
if __name__ == "__main__":
    model, history, test_acc = main()
    
    # 可选：尝试模型集成以获得更好的性能
    print("\n" + "="*50)
    print("可选：训练多个模型进行集成")
    
    try:
        # 训练多个不同的模型
        ensemble_models = []
        num_models = 3  # 训练3个不同的模型进行集成
        
        for i in range(num_models):
            print(f"\n训练集成模型 {i+1}/{num_models}")
            set_seed(42 + i * 100)  # 不同的随机种子
            
            # 每个模型使用不同的dropout率
            model_i = ImprovedFashionCNN(dropout_rate=0.2 + i*0.05).to(device)
            
            # 使用更少的epochs用于集成训练
            _, _, _, _, model_i_acc = train_model(
                model_i, train_dataloader, val_dataloader, test_dataloader, 
                device, epochs=30  # 较少的epochs用于集成
            )
            ensemble_models.append(model_i)
            print(f"模型{i+1}准确率: {model_i_acc:.2f}%")
        
        # 创建集成模型
        ensemble = EnsembleModel(ensemble_models).to(device)
        ensemble.eval()
        
        # 测试集成模型
        ensemble_preds, ensemble_targets = test_model(ensemble, test_dataloader, device)
        ensemble_acc = accuracy_score(ensemble_targets, ensemble_preds) * 100
        
        print(f"\n集成模型测试准确率: {ensemble_acc:.2f}%")
        print(f"比单个模型提升: {ensemble_acc - test_acc:.2f}%")
        
        # 保存集成模型
        torch.save({
            'ensemble_models': [m.state_dict() for m in ensemble_models],
            'ensemble_acc': ensemble_acc
        }, 'ensemble_model.pth')
        
    except Exception as e:
        print(f"\n集成模型训练失败: {e}")
        print("继续使用单个模型")
    
    print("\n" + "="*50)
    print("训练完成！")
    print("生成的文件:")
    print("  - best_model.pth (最佳模型)")
    print("  - final_model.pth (最终模型)")
    print("  - ensemble_model.pth (集成模型, 如果成功)")
    print("  - training_history.png (训练历史图)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - misclassified_examples.png (错误分类示例)")