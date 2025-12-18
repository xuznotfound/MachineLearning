# 作业三：预训练模型在CIFAR-10上的微调实验
# 运行环境：4090 GPU，PyTorch
#python anime_face_gan/train_anime_dcgan.py --batch-size 256 --workers 16 --epochs 50 --sample-interval 200

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 定义数据预处理
transform_train = transforms.Compose([
    transforms.Resize(224),  # 将CIFAR-10的32x32上采样到224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 加载CIFAR-10数据集
print("加载CIFAR-10数据集...")
batch_size = 128

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")

# 定义训练和评估函数
def train_model(model, trainloader, testloader, optimizer, criterion, num_epochs=10, model_name="Model"):
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(trainloader, desc='训练', unit='batch') as pbar:
            for inputs, labels in pbar:
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
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # 测试阶段
        test_acc = evaluate_model(model, testloader)
        test_accuracies.append(test_acc)
        
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'测试准确率: {test_acc:.2f}%')
    
    return train_losses, test_accuracies

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(testloader, desc='测试', unit='batch', leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Acc': f'{100.*correct/total:.2f}%'
                })
    
    return 100. * correct / total

# 加载预训练模型的函数
def load_pretrained_model(model_name, num_classes=10):
    if model_name == 'alexnet':
        # AlexNet模型
        model = models.alexnet(pretrained=False)
        # 修改分类器以适应CIFAR-10
        model.classifier[6] = nn.Linear(4096, num_classes)
        # 加载预训练权重
        pretrained_path = '../datasets-readonly/alexnet_pretrained_pytorch.pth'
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"已加载AlexNet预训练权重")
        else:
            print(f"警告: {pretrained_path} 不存在，使用随机初始化")
            
    elif model_name == 'vgg16':
        # VGG16模型
        model = models.vgg16(pretrained=False)
        # 修改分类器以适应CIFAR-10
        model.classifier[6] = nn.Linear(4096, num_classes)
        # 加载预训练权重
        pretrained_path = '../datasets-readonly/vgg16_pretrained_pytorch.pth'
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"已加载VGG16预训练权重")
        else:
            print(f"警告: {pretrained_path} 不存在，使用随机初始化")
            
    elif model_name == 'resnet18':
        # ResNet18模型
        model = models.resnet18(pretrained=False)
        # 修改最后的全连接层以适应CIFAR-10
        model.fc = nn.Linear(512, num_classes)
        # 加载预训练权重
        pretrained_path = '../datasets-readonly/resnet18_pretrained_pytorch.pth'
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"已加载ResNet18预训练权重")
        else:
            print(f"警告: {pretrained_path} 不存在，使用随机初始化")
            
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    return model.to(device)

# 实验1: 同时微调主干网络和分类头
def experiment_full_finetune(model_name, num_epochs=5):
    print(f"\n{'='*60}")
    print(f"实验: {model_name} - 微调整个网络")
    print('='*60)
    
    # 加载模型
    model = load_pretrained_model(model_name)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 训练模型
    start_time = time.time()
    train_losses, test_accuracies = train_model(
        model, trainloader, testloader, optimizer, criterion, num_epochs, model_name)
    end_time = time.time()
    
    print(f"\n训练时间: {end_time-start_time:.2f}秒")
    final_acc = evaluate_model(model, testloader)
    print(f"最终测试准确率: {final_acc:.2f}%")
    
    return train_losses, test_accuracies, final_acc, model

# 实验2: 固定主干网络，只微调分类头
def experiment_frozen_backbone(model_name, num_epochs=5):
    print(f"\n{'='*60}")
    print(f"实验: {model_name} - 只微调分类头")
    print('='*60)
    
    # 加载模型
    model = load_pretrained_model(model_name)
    
    # 冻结主干网络参数
    if model_name == 'alexnet':
        for param in model.features.parameters():
            param.requires_grad = False
        # 只训练分类器
        trainable_params = model.classifier.parameters()
        
    elif model_name == 'vgg16':
        for param in model.features.parameters():
            param.requires_grad = False
        # 只训练分类器
        trainable_params = model.classifier.parameters()
        
    elif model_name == 'resnet18':
        for param in model.parameters():
            param.requires_grad = False
        # 只训练最后的全连接层
        trainable_params = model.fc.parameters()
    
    # 定义损失函数和优化器（只优化可训练参数）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.001, weight_decay=1e-4)
    
    # 统计可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params_count:,} ({100.*trainable_params_count/total_params:.2f}%)")
    
    # 训练模型
    start_time = time.time()
    train_losses, test_accuracies = train_model(
        model, trainloader, testloader, optimizer, criterion, num_epochs, model_name)
    end_time = time.time()
    
    print(f"\n训练时间: {end_time-start_time:.2f}秒")
    final_acc = evaluate_model(model, testloader)
    print(f"最终测试准确率: {final_acc:.2f}%")
    
    return train_losses, test_accuracies, final_acc, model

# 主实验函数
def run_experiments():
    models_list = ['alexnet', 'vgg16', 'resnet18']
    results = {}
    
    for model_name in models_list:
        print(f"\n{'#'*80}")
        print(f"开始 {model_name} 模型的实验")
        print('#'*80)
        
        # 实验1: 微调整个网络
        print(f"\n1. {model_name} - 微调整个网络")
        train_loss_full, test_acc_full, final_acc_full, model_full = experiment_full_finetune(model_name)
        
        # 实验2: 只微调分类头
        print(f"\n2. {model_name} - 只微调分类头")
        train_loss_frozen, test_acc_frozen, final_acc_frozen, model_frozen = experiment_frozen_backbone(model_name)
        
        # 保存结果
        results[model_name] = {
            'full_finetune': {
                'train_loss': train_loss_full,
                'test_acc': test_acc_full,
                'final_acc': final_acc_full
            },
            'frozen_backbone': {
                'train_loss': train_loss_frozen,
                'test_acc': test_acc_frozen,
                'final_acc': final_acc_frozen
            }
        }
        
        # 绘制对比图
        plot_comparison(model_name, train_loss_full, test_acc_full, 
                       train_loss_frozen, test_acc_frozen)
    
    return results

# 绘制结果对比图
def plot_comparison(model_name, train_loss_full, test_acc_full, 
                   train_loss_frozen, test_acc_frozen):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_loss_full) + 1)
    
    # 绘制训练损失
    ax1.plot(epochs, train_loss_full, 'b-', label='微调整个网络', linewidth=2)
    ax1.plot(epochs, train_loss_frozen, 'r--', label='只微调分类头', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('训练损失')
    ax1.set_title(f'{model_name} - 训练损失对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制测试准确率
    ax2.plot(epochs, test_acc_full, 'b-', label='微调整个网络', linewidth=2)
    ax2.plot(epochs, test_acc_frozen, 'r--', label='只微调分类头', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('测试准确率 (%)')
    ax2.set_title(f'{model_name} - 测试准确率对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# 汇总和展示所有结果
def summarize_results(results):
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    
    summary_table = []
    
    for model_name, result in results.items():
        full_acc = result['full_finetune']['final_acc']
        frozen_acc = result['frozen_backbone']['final_acc']
        
        summary_table.append({
            '模型': model_name.upper(),
            '微调整个网络准确率 (%)': f"{full_acc:.2f}",
            '只微调分类头准确率 (%)': f"{frozen_acc:.2f}",
            '差异 (%)': f"{(full_acc - frozen_acc):.2f}",
            '推荐方法': '微调整个网络' if full_acc > frozen_acc else '只微调分类头'
        })
    
    # 打印表格
    print("\n性能对比:")
    print("-" * 70)
    print(f"{'模型':<15} {'微调整个网络':<20} {'只微调分类头':<20} {'差异':<10} {'推荐方法':<15}")
    print("-" * 70)
    
    for row in summary_table:
        print(f"{row['模型']:<15} {row['微调整个网络准确率 (%)']:<20} {row['只微调分类头准确率 (%)']:<20} {row['差异']:<10} {row['推荐方法']:<15}")
    
    # 绘制总对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = [row['模型'] for row in summary_table]
    full_accs = [float(row['微调整个网络准确率 (%)']) for row in summary_table]
    frozen_accs = [float(row['只微调分类头准确率 (%)']) for row in summary_table]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, full_accs, width, label='微调整个网络', color='blue', alpha=0.8)
    ax.bar(x + width/2, frozen_accs, width, label='只微调分类头', color='red', alpha=0.8)
    
    ax.set_xlabel('模型')
    ax.set_ylabel('测试准确率 (%)')
    ax.set_title('不同模型和微调策略的性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (full, frozen) in enumerate(zip(full_accs, frozen_accs)):
        ax.text(i - width/2, full + 0.5, f'{full:.1f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, frozen + 0.5, f'{frozen:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 分析结论
    print("\n" + "="*80)
    print("实验结论分析")
    print("="*80)
    
    print("\n1. 性能分析:")
    for row in summary_table:
        model = row['模型']
        diff = float(row['差异'])
        
        if diff > 2:
            print(f"  - {model}: 微调整个网络显著优于只微调分类头 (差异: {diff:.2f}%)")
        elif diff > 0:
            print(f"  - {model}: 微调整个网络略优于只微调分类头 (差异: {diff:.2f}%)")
        elif diff < -2:
            print(f"  - {model}: 只微调分类头显著优于微调整个网络 (差异: {-diff:.2f}%)")
        else:
            print(f"  - {model}: 两种方法性能相近 (差异: {abs(diff):.2f}%)")
    
    print("\n2. 训练效率:")
    print("  - 只微调分类头: 训练参数少，收敛快，计算成本低")
    print("  - 微调整个网络: 训练参数多，可能需要更多epoch，但可能达到更高性能")
    
    print("\n3. 推荐策略:")
    print("  - 如果数据集与预训练数据集相似: 建议只微调分类头")
    print("  - 如果数据集与预训练数据集差异较大: 建议微调整个网络")
    print("  - 如果计算资源有限: 建议只微调分类头")
    print("  - 如果追求最佳性能: 建议微调整个网络")

# 运行实验
if __name__ == "__main__":
    print("开始作业三实验...")
    print(f"设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 运行所有实验
    results = run_experiments()
    
    # 汇总结果
    summarize_results(results)
    
    print("\n实验完成！所有结果已保存。")