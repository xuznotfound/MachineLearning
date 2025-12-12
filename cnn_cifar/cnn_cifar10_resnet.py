#!/usr/bin/env python3
"""
CNN模型用于CIFAR-10分类
使用残差连接结构（Residual block）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import mixed_precision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from datetime import datetime

AUTOTUNE = tf.data.AUTOTUNE

# 设置GPU配置（充分利用4090）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU可用: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU设置错误: {e}")

# 设置随机种子确保可复现性
tf.random.set_seed(42)
np.random.seed(42)

# 启用混合精度以充分利用Tensor Core并降低显存
mixed_precision.set_global_policy("mixed_float16")

print("=" * 70)
print("CNN模型 - CIFAR-10图像分类（带残差连接）")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==================== 1. 数据加载与预处理 ====================
def load_and_preprocess_cifar10():
    """加载并预处理CIFAR-10数据集"""
    print("\n[1/6] 正在加载CIFAR-10数据集...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # 数据基本信息
    print(f"  训练集形状: {x_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集形状: {x_test.shape}, 标签: {y_test.shape}")
    print(f"  类别数: {len(np.unique(y_train))}")
    
    # 归一化像素值到[0, 1]范围
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 按通道减均值除方差（与PyTorch版本一致）
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 1, 1, 3)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # 将标签转换为one-hot编码
    y_train_onehot = keras.utils.to_categorical(y_train, 10)
    y_test_onehot = keras.utils.to_categorical(y_test, 10)
    
    # CIFAR-10类别名称
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    
    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot), class_names

# ==================== 2. 定义残差块 ====================
class ResidualBlock(layers.Layer):
    """带有跳跃连接的残差块"""
    def __init__(self, filters, kernel_size=3, stride=1, use_conv_shortcut=False,
                 kernel_regularizer=regularizers.l2(1e-4), **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_conv_shortcut = use_conv_shortcut
        self.kernel_regularizer = kernel_regularizer
        
        # 第一个卷积层
        self.conv1 = layers.Conv2D(
            filters, kernel_size, strides=stride, 
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=self.kernel_regularizer
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # 第二个卷积层
        self.conv2 = layers.Conv2D(
            filters, kernel_size, strides=1, 
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=self.kernel_regularizer
        )
        self.bn2 = layers.BatchNormalization()
        
        # 如果需要，为快捷路径添加卷积层
        if use_conv_shortcut or stride != 1:
            self.conv_shortcut = layers.Conv2D(
                filters, 1, strides=stride, 
                padding='same', kernel_initializer='he_normal',
                kernel_regularizer=self.kernel_regularizer
            )
            self.bn_shortcut = layers.BatchNormalization()
        else:
            self.conv_shortcut = None
        
        self.add = layers.Add()
        self.relu_out = layers.ReLU()
    
    def call(self, inputs, training=False):
        # 主路径
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # 快捷路径
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut, training=training)
        else:
            shortcut = inputs
        
        # 添加跳跃连接
        x = self.add([x, shortcut])
        x = self.relu_out(x)
        return x

# ==================== 3. 构建带残差连接的CNN模型 ====================
def build_residual_cnn(input_shape=(32, 32, 3), num_classes=10):
    """构建带残差连接的CNN模型"""
    inputs = keras.Input(shape=input_shape)
    
    # 初始卷积层（对齐PyTorch的通道数）
    x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 残差块组1（64通道）
    x = ResidualBlock(64, stride=1, use_conv_shortcut=False)(x)
    x = ResidualBlock(64, stride=1, use_conv_shortcut=False)(x)
    
    # 残差块组2（128通道，下采样）
    x = ResidualBlock(128, stride=2, use_conv_shortcut=True)(x)
    x = ResidualBlock(128, stride=1, use_conv_shortcut=False)(x)
    
    # 残差块组3（256通道，下采样）
    x = ResidualBlock(256, stride=2, use_conv_shortcut=True)(x)
    x = ResidualBlock(256, stride=1, use_conv_shortcut=False)(x)
    
    # 残差块组4（512通道，下采样）
    x = ResidualBlock(512, stride=2, use_conv_shortcut=True)(x)
    x = ResidualBlock(512, stride=1, use_conv_shortcut=False)(x)
    
    # 池化与分类头
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    # 创建模型
    model = keras.Model(inputs, outputs, name='Residual_CNN')
    
    return model

# ==================== 4. 可视化预测结果 ====================
def visualize_predictions(model, x_test, y_true, y_pred, class_names, num_samples=12):
    """可视化模型的预测结果"""
    plt.figure(figsize=(15, 10))
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i+1)
        plt.imshow(x_test[idx])
        
        true_label = class_names[y_true[idx][0]] if len(y_true.shape) > 1 else class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"真实: {true_label}\n预测: {pred_label}", color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('CNN模型预测示例（绿色正确，红色错误）', fontsize=14)
    plt.tight_layout()
    return plt

# ==================== 5. 训练和评估CNN ====================
def train_and_evaluate_cnn():
    """训练并评估CNN模型"""
    # 加载数据
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot), class_names = load_and_preprocess_cifar10()

    # 构建tf.data数据管道，将增强放在CPU侧
    batch_size = 128
    num_train = int(0.8 * len(x_train))
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    def augment(image, label):
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)  # padding=4
        image = tf.image.random_crop(image, size=[32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train[train_idx], y_train_onehot[train_idx]))
    train_ds = train_ds.shuffle(50000).map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_train[val_idx], y_train_onehot[val_idx]))
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_onehot))
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)
    
    # 构建模型
    print("\n[2/6] 正在构建带残差连接的CNN模型...")
    model = build_residual_cnn()
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        steps_per_execution=10
    )
    
    # 打印模型架构
    model.summary()
    
    # 设置回调函数
    def step_decay(epoch, lr):
        # 模拟PyTorch StepLR: 每50轮将学习率减半
        return lr * 0.5 if epoch > 0 and epoch % 50 == 0 else lr

    callbacks = [
        keras.callbacks.LearningRateScheduler(step_decay, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            '../models/best_cnn_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print("\n[3/6] 开始训练CNN模型...")
    print(f"  使用批量大小: {batch_size} (4090 GPU优化)")
    print(f"  训练样本数: {len(train_idx)}, 验证样本数: {len(val_idx)}")
    
    history = model.fit(
        train_ds,
        epochs=200,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    # ==================== 6. 可视化训练历史 ====================
    print("\n[4/6] 生成训练历史可视化...")
    
    # 创建图表目录
    os.makedirs('../results', exist_ok=True)
    
    plt.figure(figsize=(14, 5))
    
    # 准确率图表
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('CNN训练历史 - 准确率', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失图表
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失', linewidth=2)
    plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.title('CNN训练历史 - 损失', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==================== 7. 在测试集上评估模型 ====================
    print("\n[5/6] 在测试集上评估模型...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"  测试集损失: {test_loss:.4f}")
    print(f"  测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 预测测试集
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_flat = y_test.flatten()
    
    # 分类报告
    print("\n  分类报告:")
    print(classification_report(y_test_flat, y_pred_classes, target_names=class_names))
    
    # ==================== 8. 混淆矩阵 ====================
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_flat, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==================== 9. 可视化预测结果 ====================
    print("\n[6/6] 生成预测结果可视化...")
    fig = visualize_predictions(model, x_test, y_test, y_pred_classes, class_names)
    fig.savefig('../results/cnn_predictions_example.png', dpi=150, bbox_inches='tight')
    fig.show()
    
    # 保存最终模型
    model.save('../models/cnn_final_model.h5')
    
    print(f"\n{'='*70}")
    print(f"CNN模型训练完成!")
    print(f"最终测试准确率: {test_accuracy*100:.2f}%")
    print(f"模型已保存到: ../models/")
    print(f"可视化结果已保存到: ../results/")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    return test_accuracy

# ==================== 主程序 ====================
if __name__ == "__main__":
    try:
        accuracy = train_and_evaluate_cnn()
        print(f"\nCNN作业完成！测试准确率: {accuracy*100:.2f}%")
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()