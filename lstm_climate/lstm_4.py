#!/usr/bin/env python3
"""
最终版 LSTM 模型 - 德里气候温度预测
已修复数据泄露并改进特征工程
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
import pickle
from datetime import datetime

# 设置GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("最终版 LSTM 模型 - 德里气候温度预测")
print("修复内容：数据泄露、特征工程、时间序列问题")
print("=" * 70)

# 1. 数据加载和预处理
def load_and_preprocess():
    """加载数据"""
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

# 2. 创建时间序列（修复数据泄露）
def create_enhanced_sequences(df, window_size=14):
    """创建时间序列（修复反向填充泄露）"""
    
    # 基础特征
    base_features = ['humidity', 'meanpressure', 'wind_speed']
    
    # 创建新特征
    df_features = df.copy()
    
    # 季节性特征
    df_features['month'] = df_features['date'].dt.month
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    df_features['day_of_month'] = df_features['date'].dt.day
    df_features['day_of_week'] = df_features['date'].dt.dayofweek
    
    # 温度滞后特征
    df_features['temp_lag1'] = df_features['meantemp'].shift(1)
    df_features['temp_lag2'] = df_features['meantemp'].shift(2)
    df_features['temp_lag7'] = df_features['meantemp'].shift(7)
    
    # 湿度滞后特征
    df_features['humidity_lag1'] = df_features['humidity'].shift(1)
    df_features['humidity_lag7'] = df_features['humidity'].shift(7)
    
    # 滑动窗口统计（7天平均）
    df_features['temp_rolling_mean_7'] = df_features['meantemp'].rolling(window=7, min_periods=1).mean()
    df_features['humidity_rolling_mean_7'] = df_features['humidity'].rolling(window=7, min_periods=1).mean()
    df_features['wind_rolling_mean_7'] = df_features['wind_speed'].rolling(window=7, min_periods=1).mean()
    
    # 滑动窗口统计（3天平均）
    df_features['temp_rolling_mean_3'] = df_features['meantemp'].rolling(window=3, min_periods=1).mean()
    
    # 温度变化率
    df_features['temp_change_rate'] = df_features['meantemp'].diff()
    
    # 季节正弦/余弦编码
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year']/365)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year']/365)
    
    # 修复：仅使用前向填充，避免未来信息泄露
    df_features = df_features.fillna(method='ffill')
    
    # 最终特征集（包括所有工程特征）
    features = base_features + [
        'month', 'day_of_year', 'day_of_month', 'day_of_week',
        'temp_lag1', 'temp_lag2', 'temp_lag7',
        'humidity_lag1', 'humidity_lag7',
        'temp_rolling_mean_7', 'humidity_rolling_mean_7', 'wind_rolling_mean_7',
        'temp_rolling_mean_3', 'temp_change_rate',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos'
    ]
    
    print(f"使用特征数量: {len(features)}")
    print("特征类别:")
    print(f"  - 基础特征: {len(base_features)} 个")
    print("  - 时间特征: 4 个")
    print("  - 滞后特征: 5 个")
    print("  - 滑动统计: 4 个")
    print("  - 变化率: 1 个")
    print("  - 季节编码: 4 个")
    
    # 创建序列
    X, y = [], []
    for i in range(window_size, len(df_features)):
        X.append(df_features[features].iloc[i-window_size:i].values)
        y.append(df_features['meantemp'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n生成序列: X={X.shape}, y={y.shape}")
    print(f"每个样本: {window_size} 天 × {len(features)} 个特征")
    
    return X, y, features

# 3. LSTM模型架构
def build_lstm_model(input_shape):
    """构建LSTM模型"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # 第一层LSTM
        layers.LSTM(128, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第二层LSTM
        layers.LSTM(64, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第三层LSTM
        layers.LSTM(32, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 全连接层
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 输出层
    ])
    
    # 自适应学习率优化器
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', keras.metrics.RootMeanSquaredError()]
    )
    
    return model

# 4. 训练流程
def train_lstm_model():
    """训练LSTM模型"""
    
    # 加载数据
    print("\n[1/5] 正在加载数据...")
    combined, train_size = load_and_preprocess()
    
    # 创建序列
    print("\n[2/5] 正在创建时间序列...")
    window_size = 21  # 增加到21天窗口（建议的改进）
    X, y, features = create_enhanced_sequences(combined, window_size)
    
    # 修复：正确的训练集/测试集划分
    # 训练集：前train_size天的数据（减去窗口大小用于序列）
    # 测试集：从train_size开始到最后
    X_train = X[:train_size - window_size]
    y_train = y[:train_size - window_size]
    X_test = X[train_size - window_size:]  # 保留窗口上下文
    y_test = y[train_size - window_size:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape}（{len(X_train)} 个样本）")
    print(f"  测试集: {X_test.shape}（{len(X_test)} 个样本）")
    
    # 归一化（仅在训练集上拟合）
    print("\n[3/5] 正在归一化数据...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 重塑数据进行归一化
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 构建模型
    print("\n[4/5] 正在构建 LSTM 模型...")
    input_shape = (window_size, len(features))
    model = build_lstm_model(input_shape)
    model.summary()
    
    # 创建目录
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # 回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            '../models/lstm_best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='../logs/lstm',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # 训练模型（修复：shuffle=False避免时间泄露）
    print("\n[5/5] 开始训练模型...")
    print(f"  批大小: 32")
    print(f"  最大轮数: 200")
    print(f"  验证集比例: 20%（训练集末尾 20% 数据）")
    print(f"  是否打乱: False（保持时间顺序）")
    
    history = model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=32,
        epochs=200,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=False,  # 关键修复：时间序列不打乱
        verbose=1
    )
    
    return model, history, scaler_X, scaler_y, X_test_scaled, y_test_scaled, y_test

# 5. 评估和可视化
def evaluate_model(model, history, scaler_X, scaler_y, X_test_scaled, y_test_scaled, y_test_original):
    """评估模型并生成可视化"""
    
    # 预测
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    
    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n" + "="*60)
    print("模型评估结果")
    print("="*60)
    print(f"MSE:  {mse:.4f} (°C)²")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"MAE:  {mae:.4f} °C")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("="*60)
    
    # 保存归一化器
    with open('../models/scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('../models/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    print(f"\n已保存归一化器到 ../models/scaler_X.pkl 和 scaler_y.pkl")
    
    # 可视化训练历史
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training History - Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE (°C)', fontsize=12)
    plt.title('Training History - MAE', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'LSTM Training Performance (Final MSE: {mse:.2f})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../results/lstm_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 预测结果可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. 预测对比
    axes[0, 0].plot(y_test[:100], label='True Temperature', linewidth=2, color='blue')
    axes[0, 0].plot(y_pred[:100], label='Predicted Temperature', linewidth=2, color='red')
    axes[0, 0].set_xlabel('Sample Index', fontsize=12)
    axes[0, 0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0, 0].set_title('Prediction Comparison (First 100 Samples)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 散点图
    axes[0, 1].scatter(y_test, y_pred, alpha=0.5, s=10, color='green')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('True Temperature (°C)', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Temperature (°C)', fontsize=12)
    axes[0, 1].set_title('True vs Predicted Values', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 误差分布
    errors = y_test - y_pred
    axes[0, 2].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Prediction Error (°C)', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title(f'Error Distribution (Mean={errors.mean():.2f}°C)', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 残差图
    axes[1, 0].scatter(y_pred, errors, alpha=0.5, s=10, color='purple')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Temperature (°C)', fontsize=12)
    axes[1, 0].set_ylabel('Residuals (True - Predicted)', fontsize=12)
    axes[1, 0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 累积误差
    cumulative_error = np.cumsum(np.abs(errors))
    axes[1, 1].plot(cumulative_error, linewidth=2, color='brown')
    axes[1, 1].set_xlabel('Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative Absolute Error (°C)', fontsize=12)
    axes[1, 1].set_title('Cumulative Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 性能指标
    metrics = ['MSE', 'RMSE', 'MAE', 'R²']
    values = [mse, rmse, mae, r2]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    bars = axes[1, 2].bar(metrics, values, color=colors, edgecolor='black')
    axes[1, 2].set_ylabel('Value', fontsize=12)
    axes[1, 2].set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, axis='y', alpha=0.3)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metrics[bars.index(bar)] == 'R²':
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'LSTM Model Performance Analysis (Final R²: {r2:.3f})', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../results/lstm_performance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'True_Temperature': y_test,
        'Predicted_Temperature': y_pred,
        'Error': errors,
        'Absolute_Error': np.abs(errors),
        'Percentage_Error': np.abs(errors / y_test) * 100
    })
    
    results_df.to_csv('../results/lstm_detailed_predictions.csv', index=False)
    
    # 生成性能报告
    print("\n" + "="*60)
    print("性能报告摘要")
    print("="*60)
    
    # 性能评级
    if mse < 3:
        rating = "表现优秀"
        advice = "模型对温度预测表现极佳。"
    elif mse < 6:
        rating = "表现良好"
        advice = "模型达到实用需求。"
    elif mse < 10:
        rating = "表现一般"
        advice = "模型可用，但仍有提升空间。"
    else:
        rating = "需要改进"
        advice = "模型需要明显改进后才能实用。"
    
    print(f"总体评价: {rating}")
    print(f"\n关键发现:")
    print(f"• 平均预测误差: {rmse:.2f}°C")
    print(f"• 模型解释温度方差的 {r2*100:.1f}%")
    print(f"• 平均绝对百分比误差: {mape:.1f}%")
    print(f"\n建议: {advice}")
    print("="*60)
    
    return mse, rmse, mae, r2, y_test, y_pred

# 主程序
if __name__ == "__main__":
    try:
        print("开始训练最终版 LSTM 模型（包含修复）...")
        print("-" * 60)
        print("修复项:")
        print("1. 去除反向填充以避免未来信息泄露")
        print("2. 训练时 shuffle=False 保持时间顺序")
        print("3. 加强特征工程")
        print("4. 时间窗口增加到 21 天")
        print("5. 保存归一化器以便复现")
        print("6. 使用 .keras 格式替代 .h5")
        print("-" * 60)
        
        # 训练模型
        model, history, scaler_X, scaler_y, X_test_scaled, y_test_scaled, y_test_original = train_lstm_model()
        
        # 评估模型
        mse, rmse, mae, r2, y_test, y_pred = evaluate_model(
            model, history, scaler_X, scaler_y, 
            X_test_scaled, y_test_scaled, y_test_original
        )
        
        # 保存最终模型
        model.save('../models/lstm_final_model.keras')
        print(f"\n最终模型已保存到: ../models/lstm_final_model.keras")
        print(f"训练曲线已保存到: ../results/lstm_training_history.png")
        print(f"性能分析图已保存到: ../results/lstm_performance_analysis.png")
        print(f"预测明细已保存到: ../results/lstm_detailed_predictions.csv")
        
        print("\n" + "="*60)
        print("训练完成 - 所有修复已应用")
        print("="*60)
        print(f"最终测试 MSE:  {mse:.4f} (°C)²")
        print(f"最终测试 R²:   {r2:.4f}")
        print(f"最终测试 RMSE: {rmse:.4f} °C")
        print("="*60)
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()