# 作业1：CIFAR-10 简单 CNN

本目录提供一个在 CIFAR-10 上训练小型卷积神经网络的完整脚本，并支持可选的残差块。

## 依赖
- Python 3.9+
- PyTorch、torchvision、numpy

在当前环境缺少依赖时先安装：
```
pip install torch torchvision numpy
```

## 运行
默认数据目录指向仓库已有的 `data`，可自动下载缺失数据。
```
python train_cifar10_cnn.py --data-dir ../data --epochs 15 --batch-size 128 --save-path outputs/simple_cnn_cifar10.pth
```
常用参数：
- `--use-residual` 启用中间的残差块（可选加分项）。
- `--cpu` 在无 GPU 或需强制 CPU 时使用。
- `--seed` 控制随机性（默认 42）。

训练结束会在 `outputs/simple_cnn_cifar10.pth` 保存最佳权重，控制台会打印每轮的训练/测试损失与准确率以及最佳测试准确率。
