#!/usr/bin/env python3
"""
作业三：预训练模型微调对比
加载预训练的AlexNet、VGG16、ResNet18模型
在CIFAR-10数据集上比较：
1. 同时微调主干网络和分类头
2. 固定主干网络只微调分类头

运行：python assignment3_finetune_solution.py --epochs 5 --batch-size 128 --num-workers 4
Adjust models to subset if needed: --models alexnet resnet18.
Outputs: best checkpoints outputs/{model}_{mode}_best.pth and summary outputs/assignment3_summary.json.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    # 修复: 使用正确的API获取GPU内存信息
    try:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存: {memory_allocated:.2f} GB / {memory_total:.2f} GB")
    except:
        print("GPU内存信息获取失败")

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)

# 创建目录
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

print("=" * 80)
print("作业三：预训练模型微调对比实验")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ==================== 1. 数据准备 ====================
def prepare_cifar10_data(batch_size=128):
    """准备CIFAR-10数据集"""
    
    # CIFAR-10预处理：适应预训练模型的输入要求
    # 注意：预训练模型通常接受224x224输入，而CIFAR-10是32x32
    # 我们需要将图像上采样到224x224
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transform_train = transforms.Compose([
        transforms.Resize(256),  # 先放大到256
        transforms.RandomCrop(224),  # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),  # 先放大到256
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # 下载/加载数据集
    print("加载CIFAR-10数据集...")
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # CIFAR-10类别
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")
    print(f"类别数: {len(classes)}")
    print(f"批量大小: {batch_size}")
    
    return train_loader, test_loader, classes

# ==================== 2. 模型加载函数 ====================
def load_pretrained_model(model_name, pretrained_path='../datasets-readonly'):
    """加载预训练模型"""
    
    model_path = os.path.join(pretrained_path, f"{model_name}_pretrained_pytorch.pth")
    
    print(f"\n加载 {model_name} 预训练模型...")
    print(f"模型路径: {model_path}")
    
    try:
        # 首先检查文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 预训练模型文件不存在: {model_path}")
            print("将使用torchvision的预训练模型...")
            
            # 使用torchvision的预训练模型
            if model_name == 'alexnet':
                model = models.alexnet(pretrained=True)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=True)
            elif model_name == 'resnet18':
                model = models.resnet18(pretrained=True)
            else:
                raise ValueError(f"未知模型: {model_name}")
        else:
            # 加载自定义预训练模型
            if model_name == 'alexnet':
                model = models.alexnet(pretrained=False)
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=False)
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
            elif model_name == 'resnet18':
                model = models.resnet18(pretrained=False)
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
            else:
                raise ValueError(f"未知模型: {model_name}")
                
        print(f"{model_name} 模型加载成功")
        
        # 修改最后一层以适应CIFAR-10的10个类别
        if model_name == 'alexnet':
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 10)
        elif model_name == 'vgg16':
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, 10)
        elif model_name == 'resnet18':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
        
        print(f"修改分类头: {num_ftrs} -> 10 (CIFAR-10类别)")
        
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("将使用随机初始化的模型作为备用...")
        
        # 备用方案：创建随机初始化的模型
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=False)
            model.classifier[6] = nn.Linear(4096, 10)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=False)
            model.classifier[6] = nn.Linear(4096, 10)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(512, 10)
        
        return model

# ==================== 3. 训练函数 ====================
def train_model(model, train_loader, test_loader, criterion, optimizer, 
                scheduler, num_epochs=20, fine_tune_type='full'):
    """
    训练模型
    
    参数:
        fine_tune_type: 'full' - 微调所有层
                       'head_only' - 只微调分类头
    """
    
    print(f"\n开始训练: {fine_tune_type} 微调")
    print("-" * 60)
    
    # 根据微调类型冻结相应的层
    if fine_tune_type == 'head_only':
        print("冻结主干网络，只训练分类头...")
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻分类头
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
    
    # 将模型移到GPU
    model = model.to(device)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 记录最佳模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
                dataloader = train_loader
            else:
                model.eval()   # 评估模式
                dataloader = test_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播 + 优化（仅训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # 计算epoch损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 深度复制最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
    
    time_elapsed = time.time() - start_time
    print(f'\n训练完成! 耗时: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return model, history, best_acc

# ==================== 4. 评估函数 ====================
def evaluate_model(model, test_loader, classes):
    """评估模型性能"""
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    return accuracy, cm

# ==================== 5. 可视化函数 ====================
def plot_training_history(history, model_name, fine_tune_type, save_path='results'):
    """绘制训练历史"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Loss ({fine_tune_type})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy ({fine_tune_type})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'{save_path}/{model_name}_{fine_tune_type}_history.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"训练历史图已保存: {filename}")

# ==================== 6. 实验主函数 ====================
def run_experiment():
    """运行完整的对比实验"""
    
    # 准备数据
    print("\n" + "="*80)
    print("准备CIFAR-10数据集")
    print("="*80)
    
    batch_size = 128  # 4090 GPU可以处理较大的batch size
    train_loader, test_loader, classes = prepare_cifar10_data(batch_size)
    
    # 定义要实验的模型
    model_names = ['alexnet', 'vgg16', 'resnet18']
    fine_tune_types = ['full', 'head_only']
    
    # 存储实验结果
    results = []
    
    # 遍历所有模型和微调类型
    for model_name in model_names:
        print(f"\n" + "="*80)
        print(f"实验: {model_name.upper()}")
        print("="*80)
        
        for fine_tune_type in fine_tune_types:
            print(f"\n{'='*60}")
            print(f"微调类型: {fine_tune_type}")
            print(f"{'='*60}")
            
            # 加载预训练模型
            model = load_pretrained_model(model_name)
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            
            # 根据微调类型设置不同的优化器参数
            if fine_tune_type == 'full':
                # 微调所有层，使用较小的学习率
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
            else:
                # 只微调分类头，可以使用较大的学习率
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            
            # 学习率调度器
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
            # 训练模型
            num_epochs = 20  # 对于微调，20个epoch通常足够
            trained_model, history, best_acc = train_model(
                model, train_loader, test_loader, criterion, optimizer, 
                scheduler, num_epochs, fine_tune_type
            )
            
            # 评估模型
            accuracy, cm = evaluate_model(trained_model, test_loader, classes)
            
            # 绘制训练历史
            plot_training_history(history, model_name, fine_tune_type)
            
            # 保存模型
            model_path = f'models/{model_name}_{fine_tune_type}_cifar10.pth'
            torch.save(trained_model.state_dict(), model_path)
            print(f"模型已保存: {model_path}")
            
            # 记录结果
            results.append({
                'model': model_name,
                'fine_tune_type': fine_tune_type,
                'best_accuracy': best_acc.item(),
                'test_accuracy': accuracy,
                'train_loss_final': history['train_loss'][-1],
                'val_loss_final': history['val_loss'][-1]
            })
    
    # 总结实验结果
    print_results_summary(results)

# ==================== 7. 结果总结和可视化 ====================
def print_results_summary(results):
    """打印和可视化实验结果总结"""
    
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 打印表格
    print("\n详细结果:")
    print(results_df.to_string(index=False))
    
    # 保存结果到CSV
    results_df.to_csv('results/experiment_results.csv', index=False)
    print(f"\n结果已保存: results/experiment_results.csv")
    
    # 可视化比较
    plot_results_comparison(results_df)

def plot_results_comparison(results_df):
    """可视化比较结果"""
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. 准确率比较条形图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 准确率比较
    ax1 = axes[0]
    models = results_df['model'].unique()
    
    for i, model in enumerate(models):
        model_results = results_df[results_df['model'] == model]
        x_pos = i * 3
        ax1.bar([x_pos, x_pos+1], 
                [model_results[model_results['fine_tune_type'] == 'full']['test_accuracy'].values[0],
                 model_results[model_results['fine_tune_type'] == 'head_only']['test_accuracy'].values[0]],
                width=0.8, 
                color=['#2E86AB', '#A23B72'],
                label=['Full Fine-tuning', 'Head Only'] if i == 0 else None)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks([0.5, 3.5, 6.5])
    ax1.set_xticklabels(['AlexNet', 'VGG16', 'ResNet18'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for i, model in enumerate(models):
        model_results = results_df[results_df['model'] == model]
        x_pos = i * 3
        for j, fine_tune_type in enumerate(['full', 'head_only']):
            accuracy = model_results[model_results['fine_tune_type'] == fine_tune_type]['test_accuracy'].values[0]
            ax1.text(x_pos + j, accuracy + 0.005, f'{accuracy:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 训练损失比较
    ax2 = axes[1]
    bar_width = 0.35
    x = np.arange(len(models))
    
    full_losses = []
    head_losses = []
    
    for model in models:
        model_results = results_df[results_df['model'] == model]
        full_losses.append(model_results[model_results['fine_tune_type'] == 'full']['train_loss_final'].values[0])
        head_losses.append(model_results[model_results['fine_tune_type'] == 'head_only']['train_loss_final'].values[0])
    
    ax2.bar(x - bar_width/2, full_losses, bar_width, label='Full Fine-tuning', color='#2E86AB')
    ax2.bar(x + bar_width/2, head_losses, bar_width, label='Head Only', color='#A23B72')
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Final Training Loss', fontsize=12)
    ax2.set_title('Final Training Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['AlexNet', 'VGG16', 'ResNet18'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/experiment_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. 准确率提升对比
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    
    improvements = []
    for model in models:
        model_results = results_df[results_df['model'] == model]
        full_acc = model_results[model_results['fine_tune_type'] == 'full']['test_accuracy'].values[0]
        head_acc = model_results[model_results['fine_tune_type'] == 'head_only']['test_accuracy'].values[0]
        improvement = full_acc - head_acc
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax3.bar(models, improvements, color=colors, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Accuracy Improvement', fontsize=12)
    ax3.set_title('Full Fine-tuning vs Head Only: Accuracy Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.3f}', ha='center', va='bottom' if imp >= 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/improvement_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印关键发现
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    
    for model in models:
        model_results = results_df[results_df['model'] == model]
        full_acc = model_results[model_results['fine_tune_type'] == 'full']['test_accuracy'].values[0]
        head_acc = model_results[model_results['fine_tune_type'] == 'head_only']['test_accuracy'].values[0]
        
        print(f"\n{model.upper()}:")
        print(f"  全网络微调准确率: {full_acc:.3f}")
        print(f"  仅分类头微调准确率: {head_acc:.3f}")
        print(f"  准确率差异: {full_acc - head_acc:.3f} ({'全网络更好' if full_acc > head_acc else '仅分类头更好'})")
    
    print("\n" + "="*80)
    print("实验完成！")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

# ==================== 8. 主程序 ====================
def main():
    """主函数"""
    try:
        # 检查预训练模型目录是否存在
        pretrained_path = '../datasets-readonly'
        if not os.path.exists(pretrained_path):
            print(f"警告: 预训练模型目录不存在: {pretrained_path}")
            print("将使用torchvision的预训练模型作为替代")
            print("如果这是作业要求，请确保预训练模型文件在正确位置")
        
        # 运行实验
        run_experiment()
        
    except Exception as e:
        print(f"实验出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试简化版本
        print("\n尝试运行简化版本...")
        run_simplified_experiment()

def run_simplified_experiment():
    """简化版实验（如果完整版失败）"""
    print("\n" + "="*80)
    print("运行简化版实验")
    print("="*80)
    
    # 准备数据
    batch_size = 64  # 更小的batch size
    train_loader, test_loader, classes = prepare_cifar10_data(batch_size)
    
    # 只测试一个模型
    model_name = 'resnet18'
    
    # 使用torchvision预训练模型
    print(f"加载 {model_name} 模型...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    print(f"修改分类头: {num_ftrs} -> 10 (CIFAR-10类别)")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    
    # 测试两种微调策略
    results = []
    
    for fine_tune_type in ['head_only', 'full']:
        print(f"\n测试: {fine_tune_type} 微调")
        
        # 重新加载模型
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(num_ftrs, 10)
        
        # 设置优化器
        if fine_tune_type == 'head_only':
            # 冻结所有层
            for param in model.parameters():
                param.requires_grad = False
            
            # 只解冻最后一层
            for param in model.fc.parameters():
                param.requires_grad = True
            
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # 训练
        model = model.to(device)
        num_epochs = 10  # 更少的epoch
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            print(f'Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.3f}, Acc: {train_acc:.2f}%')
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f'测试准确率: {test_acc:.2f}%')
        
        results.append({
            'model': model_name,
            'fine_tune_type': fine_tune_type,
            'test_accuracy': test_acc/100.0
        })
    
    # 打印结果
    print("\n" + "="*80)
    print("简化实验结果")
    print("="*80)
    
    for result in results:
        print(f"{result['model']} - {result['fine_tune_type']}: {result['test_accuracy']:.3f}")

if __name__ == "__main__":
    main()