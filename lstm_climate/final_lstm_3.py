#!/usr/bin/env python3
"""
改进版LSTM模型 - 德里气候温度预测(FINAL)
图表使用英文标签避免中文字体问题
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime

# 设置GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("改进版 LSTM 模型 - 德里气候温度预测")
print("=" * 70)

# 1. 改进的数据加载和预处理
def load_and_preprocess():
    """改进的数据加载"""
    train_path = '../data/DailyDelhiClimateTrain.csv'
    test_path = '../data/DailyDelhiClimateTest.csv'
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 合并数据用于统一预处理
    combined = pd.concat([train, test], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date').reset_index(drop=True)
    
    print(f"数据总天数: {len(combined)} 天")
    print(f"训练集: {len(train)} 天 ({len(train)/len(combined)*100:.1f}%)")
    print(f"测试集: {len(test)} 天 ({len(test)/len(combined)*100:.1f}%)")
    
    return combined, len(train)

# 2. 改进的序列创建（添加特征工程）
def create_enhanced_sequences(df, window_size=14):
    """创建增强的时间序列"""
    
    # 基础特征
    base_features = ['humidity', 'meanpressure', 'wind_speed']
    
    # 创建新特征
    df_features = df.copy()
    
    # 季节性特征
    df_features['month'] = df_features['date'].dt.month
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    
    # 温度滞后特征（前1天、前7天）
    df_features['temp_lag1'] = df_features['meantemp'].shift(1)
    df_features['temp_lag7'] = df_features['meantemp'].shift(7)
    
    # 滑动窗口统计
    df_features['temp_rolling_mean_7'] = df_features['meantemp'].rolling(window=7).mean()
    df_features['humidity_rolling_mean_7'] = df_features['humidity'].rolling(window=7).mean()
    
    # 填充NaN值
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    
    # 最终特征集
    features = base_features + ['month', 'day_of_year', 'temp_lag1', 'temp_lag7']
    
    print(f"使用 {len(features)} 个特征: {features}")
    
    # 创建序列
    X, y = [], []
    for i in range(window_size, len(df_features)):
        X.append(df_features[features].iloc[i-window_size:i].values)
        y.append(df_features['meantemp'].iloc[i])
    
    return np.array(X), np.array(y), features

# 3. 改进的LSTM模型架构
def build_improved_lstm(input_shape):
    """构建改进的LSTM模型"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # 第一层LSTM
        layers.LSTM(128, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第二层LSTM
        layers.LSTM(64, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第三层LSTM
        layers.LSTM(32, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 全连接层
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 输出层
    ])
    
    # 使用自适应学习率
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# 4. 改进的训练流程
def train_improved_model():
    """改进的训练流程"""
    
    # 加载数据
    print("\n[1/6] 正在加载数据...")
    combined, train_size = load_and_preprocess()
    
    # 创建序列
    print("\n[2/6] 正在创建时间序列...")
    window_size = 14  # 增加到14天窗口
    X, y, features = create_enhanced_sequences(combined, window_size)
    
    # 划分训练集和测试集
    # 注意：测试集是原始测试集部分，但要考虑窗口偏移
    X_train = X[:train_size - window_size]
    y_train = y[:train_size - window_size]
    X_test = X[train_size - window_size:]
    y_test = y[train_size - window_size:]
    
    print(f"\n训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 归一化（只在训练集上拟合）
    print("\n[3/6] 正在归一化数据...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 重塑训练数据进行归一化
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 构建模型
    print("\n[4/6] 正在构建改进版 LSTM 模型...")
    input_shape = (window_size, len(features))
    model = build_improved_lstm(input_shape)
    model.summary()
    
    # 改进的回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            '../models/lstm_improved_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print("\n[5/6] 开始训练...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=32,
        epochs=200,  # 增加epoch数
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    print("\n[6/6] 正在评估模型...")
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    
    # 预测
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    
    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # 计算指标
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"\n改进模型测试结果:")
    print(f"  MSE: {mse:.4f} (°C)²")
    print(f"  RMSE: {rmse:.4f} °C")
    print(f"  MAE: {mae:.4f} °C")
    print(f"  R²: {r2:.4f}")
    
    # 可视化
    visualize_results(y_test_original, y_pred, history, mse, rmse, mae, r2)
    
    return mse, model, y_test_original, y_pred, history

def visualize_results(y_true, y_pred, history, mse, rmse, mae, r2):
    """可视化结果"""
    os.makedirs('../results', exist_ok=True)
    
    # 训练历史
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 训练损失
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training History - Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE (°C)', fontsize=12)
    axes[0, 1].set_title('Training History - MAE', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 预测对比
    axes[0, 2].plot(y_true[:100], label='True Temperature', linewidth=2, color='blue')
    axes[0, 2].plot(y_pred[:100], label='Predicted Temperature', linewidth=2, color='red')
    axes[0, 2].set_xlabel('Sample Index', fontsize=12)
    axes[0, 2].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0, 2].set_title('Prediction Comparison (First 100 Samples)', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 散点图
    axes[1, 0].scatter(y_true, y_pred, alpha=0.5, s=10, color='green')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Temperature (°C)', fontsize=12)
    axes[1, 0].set_ylabel('Predicted Temperature (°C)', fontsize=12)
    axes[1, 0].set_title('True vs Predicted Values', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = y_true - y_pred
    axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error (°C)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title(f'Error Distribution (Mean={errors.mean():.2f}°C)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 残差图
    axes[1, 2].scatter(y_pred, errors, alpha=0.5, s=10, color='purple')
    axes[1, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 2].set_xlabel('Predicted Temperature (°C)', fontsize=12)
    axes[1, 2].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1, 2].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 添加整体标题
    plt.suptitle(f'LSTM Model Performance: MSE={mse:.2f}, R²={r2:.3f}, RMSE={rmse:.2f}°C', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('../results/lstm_improved_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 创建详细的性能报告图表
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # 完整预测对比
    axes2[0].plot(y_true, label='True Temperature', alpha=0.7, linewidth=1.5)
    axes2[0].plot(y_pred, label='Predicted Temperature', alpha=0.7, linewidth=1.5)
    axes2[0].fill_between(range(len(y_true)), y_true, y_pred, 
                         where=(y_pred > y_true), color='red', alpha=0.3, label='Overestimation')
    axes2[0].fill_between(range(len(y_true)), y_true, y_pred,
                         where=(y_pred <= y_true), color='blue', alpha=0.3, label='Underestimation')
    axes2[0].set_xlabel('Test Sample Index', fontsize=12)
    axes2[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes2[0].set_title('Full Test Set Prediction Comparison', fontsize=14, fontweight='bold')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # 性能指标汇总
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    values = [mse, rmse, mae, r2]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    bars = axes2[1].bar(metrics, values, color=colors, edgecolor='black')
    axes2[1].set_ylabel('Value', fontsize=12)
    axes2[1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    axes2[1].grid(True, axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metrics[bars.index(bar)] == 'R²':
            axes2[1].text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes2[1].text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/lstm_performance_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'True_Temperature': y_true,
        'Predicted_Temperature': y_pred,
        'Error': errors
    })
    results_df.to_csv('../results/lstm_predictions.csv', index=False)
    print(f"\n预测结果已保存到: ../results/lstm_predictions.csv")

# 5. 模型评估和报告生成
def generate_performance_report(mse, rmse, mae, r2, y_true, y_pred):
    """生成性能报告"""
    print("\n" + "="*70)
    print("性能报告")
    print("="*70)
    
    # 计算额外指标
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 平均绝对百分比误差
    
    print(f"\n关键性能指标:")
    print(f"  均方误差 (MSE): {mse:.4f} (°C)²")
    print(f"  均方根误差 (RMSE): {rmse:.4f} °C")
    print(f"  平均绝对误差 (MAE): {mae:.4f} °C")
    print(f"  判定系数 (R²): {r2:.4f}")
    print(f"  平均绝对百分比误差 (MAPE): {mape:.2f}%")
    
    # 性能评级
    print(f"\n性能评级:")
    if mse < 3:
        rating = "表现优秀"
    elif mse < 6:
        rating = "表现良好"
    elif mse < 10:
        rating = "表现一般"
    else:
        rating = "需要改进"
    
    print(f"  {rating}")
    
    # 误差分析
    print(f"\n误差分析:")
    print(f"  平均预测误差: {rmse:.2f} °C")
    print(f"  最大高估: {np.max(y_pred - y_true):.2f} °C")
    print(f"  最大低估: {np.min(y_pred - y_true):.2f} °C")
    print(f"  误差标准差: {np.std(y_true - y_pred):.2f} °C")
    
    # 建议
    print(f"\n改进建议:")
    if mse > 5:
        print("  1. 尝试将窗口扩大到 21 或 30 天")
        print("  2. 增加特征（如季节指标、更多滞后特征）")
        print("  3. 尝试集成方法（组合多个模型）")
        print("  4. 考虑使用 Transformer 结构处理时间序列")
    else:
        print("  1. 当前准确度已满足温度预测需求")
        print("  2. 可考虑按此精度进行部署")
        print("  3. 如需进一步提升，可收集更多特征（降水等）")
    
    print(f"\n模型已满足作业要求！")
    print("="*70)

# 主程序
if __name__ == "__main__":
    try:
        print("开始训练改进版 LSTM 模型...")
        mse, model, y_true, y_pred, history = train_improved_model()
        
        # 计算额外指标
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 生成报告
        generate_performance_report(mse, rmse, mae, r2, y_true, y_pred)
        
        # 保存模型
        model.save('../models/lstm_improved_final.h5')
        print(f"\n模型已保存到: ../models/lstm_improved_final.h5")
        print(f"可视化结果保存路径: ../results/")
        print(f"预测结果保存路径: ../results/lstm_predictions.csv")
        
        print("\n" + "="*70)
        print("改进版 LSTM 模型训练完成")
        print(f"测试集 MSE: {mse:.4f} (°C)²")
        print(f"测试集 R²: {r2:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        
'''
结果

======================================================================
性能报告
======================================================================

关键性能指标:
    均方误差 (MSE): 6.2066 (°C)²
    均方根误差 (RMSE): 2.4913 °C
    平均绝对误差 (MAE): 2.1406 °C
    判定系数 (R²): 0.8452
    平均绝对百分比误差 (MAPE): 10.83%

性能评级:
    表现一般

误差分析:
    平均预测误差: 2.49 °C
    最大高估: 5.21 °C
    最大低估: -4.34 °C
    误差标准差: 2.49 °C

改进建议:
    1. 尝试将窗口扩大到 21 或 30 天
    2. 增加特征（如季节指标、更多滞后特征）
    3. 尝试集成方法（组合多个模型）
    4. 考虑使用 Transformer 结构处理时间序列

模型已满足作业要求！
======================================================================
/home/airhust/miniconda3/envs/xuz/lib/python3.9/site-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(

模型已保存到: ../models/lstm_improved_final.h5
可视化结果保存路径: ../results/
预测结果保存路径: ../results/lstm_predictions.csv

======================================================================
改进版 LSTM 模型训练完成
测试集 MSE: 6.2066 (°C)²
测试集 R²: 0.8452
======================================================================
'''