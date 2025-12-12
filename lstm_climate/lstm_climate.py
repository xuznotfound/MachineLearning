#!/usr/bin/env python3
"""
改进版LSTM模型 - 德里气候温度预测
解决中文显示问题，包含详细注释
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
import warnings
from datetime import datetime
import matplotlib

warnings.filterwarnings('ignore')

# ==================== 解决中文显示问题 ====================
# 方法1: 尝试使用系统自带的中文字体
def setup_chinese_font():
    """设置中文字体，避免警告"""
    try:
        # 列出常见的中文字体（根据系统调整）
        chinese_fonts = [
            'DejaVu Sans',  # 通常可用的字体
            'Arial Unicode MS',
            'Microsoft YaHei',
            'SimHei',
            'STHeiti',
            'WenQuanYi Micro Hei',
            'Noto Sans CJK SC',
        ]
        
        # 尝试找到可用的中文字体
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        
        for font in chinese_fonts:
            if any(font.lower() in f.lower() for f in available_fonts):
                # 设置matplotlib使用中文字体
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
                print(f"使用字体: {font}")
                return True
        
        # 如果没找到中文字体，使用默认字体，将中文标签改为英文
        print("未找到中文字体，将使用英文标签")
        return False
        
    except Exception as e:
        print(f"字体设置错误: {e}")
        return False

# 调用字体设置函数
has_chinese_font = setup_chinese_font()

# ==================== GPU配置 ====================
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

print("=" * 70)
print("改进版LSTM模型 - 德里气候温度预测")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==================== 1. 数据加载与预处理 ====================
def load_and_preprocess():
    """加载和预处理数据"""
    print("\n[1/6] 正在加载气候数据集...")
    
    train_path = '../data/DailyDelhiClimateTrain.csv'
    test_path = '../data/DailyDelhiClimateTest.csv'
    
    try:
        # 加载训练集和测试集
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        print(f"训练集: {train.shape}，测试集: {test.shape}")
        
        # 合并数据用于统一预处理
        combined = pd.concat([train, test], ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values('date').reset_index(drop=True)
        
        print(f"总数据量: {len(combined)} 天")
        print(f"训练集占比: {len(train)/len(combined)*100:.1f}%")
        print(f"测试集占比: {len(test)/len(combined)*100:.1f}%")
        
        return combined, len(train)
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, 0

# ==================== 2. 特征工程和序列创建 ====================
def create_enhanced_sequences(df, window_size=14):
    """创建增强的时间序列特征"""
    
    print(f"\n[2/6] 创建时间序列 (窗口大小={window_size})...")
    
    # 基础特征：湿度、气压、风速
    base_features = ['humidity', 'meanpressure', 'wind_speed']
    
    # 创建特征副本
    df_features = df.copy()
    
    # ===== 特征工程 =====
    # 1. 时间特征
    df_features['month'] = df_features['date'].dt.month
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    df_features['day_of_week'] = df_features['date'].dt.dayofweek
    
    # 2. 滞后特征 (前1天、前7天温度)
    df_features['temp_lag1'] = df_features['meantemp'].shift(1)
    df_features['temp_lag7'] = df_features['meantemp'].shift(7)
    
    # 3. 滑动窗口统计
    df_features['temp_rolling_mean_7'] = df_features['meantemp'].rolling(window=7, min_periods=1).mean()
    df_features['humidity_rolling_mean_7'] = df_features['humidity'].rolling(window=7, min_periods=1).mean()
    
    # 4. 差值特征
    df_features['temp_diff_1'] = df_features['meantemp'].diff(1)
    df_features['humidity_diff_1'] = df_features['humidity'].diff(1)
    
    # 填充缺失值（由shift和rolling产生）
    df_features = df_features.fillna(method='bfill').fillna(method='ffill')
    
    # 最终特征集（选择最重要的特征）
    features = base_features + [
        'month', 'day_of_year', 
        'temp_lag1', 'temp_lag7',
        'temp_rolling_mean_7'
    ]
    
    print(f"使用 {len(features)} 个特征")
    print(f"特征列表: {features}")
    
    # 创建时间序列
    X, y = [], []
    for i in range(window_size, len(df_features)):
        X.append(df_features[features].iloc[i-window_size:i].values)
        y.append(df_features['meantemp'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"序列形状: X={X.shape}, y={y.shape}")
    print(f"每个样本: {window_size}天 × {len(features)}个特征")
    
    return X, y, features

# ==================== 3. 构建改进的LSTM模型 ====================
def build_improved_lstm(input_shape):
    """构建改进的LSTM模型架构"""
    
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
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第三层LSTM
        layers.LSTM(32, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # 全连接层
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 输出层，回归问题
    ])
    
    # 优化器配置
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='mse',           # 均方误差损失
        metrics=['mae', 'mse'] # 监控指标
    )
    
    return model

# ==================== 4. 训练和评估 ====================
def train_and_evaluate_improved():
    """训练并评估改进模型"""
    
    # 加载数据
    combined, train_size = load_and_preprocess()
    if combined is None:
        return None, None
    
    # 创建序列
    window_size = 14
    X, y, features = create_enhanced_sequences(combined, window_size)
    
    # 划分训练集和测试集
    X_train = X[:train_size - window_size]
    y_train = y[:train_size - window_size]
    X_test = X[train_size - window_size:]
    y_test = y[train_size - window_size:]
    
    print(f"\n[3/6] 数据集划分:")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 数据归一化（只在训练集上拟合！）
    print("\n[4/6] 数据归一化...")
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
    print("\n[5/6] 构建改进的LSTM模型...")
    input_shape = (window_size, len(features))
    model = build_improved_lstm(input_shape)
    
    # 打印模型摘要
    model.summary()
    
    # 创建保存目录
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # 回调函数
    callbacks = [
        # 早停法
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        # 保存最佳模型
        keras.callbacks.ModelCheckpoint(
            '../models/lstm_improved_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print("\n[6/6] 开始训练模型...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=32,
        epochs=150,  # 增加epoch数
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # ==================== 5. 评估模型 ====================
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    # 在测试集上评估
    test_loss, test_mae, test_mse = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    
    # 进行预测
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    
    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    
    print(f"\n测试集评估结果:")
    print(f"MSE (均方误差): {mse:.4f} (°C)²")
    print(f"RMSE (均方根误差): {rmse:.4f} °C")
    print(f"MAE (平均绝对误差): {mae:.4f} °C")
    print(f"R² 分数: {r2:.4f}")
    
    # 评估标准
    if mse < 2:
        print("  结果: 优秀")
    elif mse < 5:
        print("  结果: 良好")
    elif mse < 10:
        print("  结果: 一般")
    else:
        print("  结果: 需要改进")
    
    # ==================== 6. 可视化结果 ====================
    visualize_results(y_test_original, y_pred, history, mse, mae, rmse, r2)
    
    # 保存模型
    model.save('../models/lstm_improved_final.h5')
    
    print(f"\n模型已保存: ../models/lstm_improved_final.h5")
    
    return mse, model

# ==================== 7. 可视化函数 ====================
def visualize_results(y_true, y_pred, history, mse, mae, rmse, r2):
    """可视化训练结果和预测"""
    
    print("\n📈 生成可视化结果...")
    
    # 根据字体支持选择标签语言
    if has_chinese_font:
        # 中文标签
        labels = {
            'loss': '损失',
            'mae': '平均绝对误差(MAE)',
            'epoch': '训练轮数(Epoch)',
            'train_loss': '训练损失',
            'val_loss': '验证损失',
            'train_mae': '训练MAE',
            'val_mae': '验证MAE',
            'true_temp': '真实温度',
            'pred_temp': '预测温度',
            'sample_index': '样本索引',
            'temperature': '温度 (°C)',
            'comparison': f'温度预测对比\nMSE={mse:.2f}, R²={r2:.2f}',
            'scatter': '真实值 vs 预测值',
            'true_temp_scatter': '真实温度 (°C)',
            'pred_temp_scatter': '预测温度 (°C)',
            'error': '预测误差',
            'error_dist': '预测误差分布',
            'error_value': '误差 (°C)',
            'frequency': '频数',
            'error_mean': f'平均误差: {mae:.2f}°C',
            'residual': '残差图',
            'pred_temp_residual': '预测温度 (°C)',
            'residual_value': '残差 (真实-预测) (°C)'
        }
    else:
        # 英文标签
        labels = {
            'loss': 'Loss',
            'mae': 'Mean Absolute Error (MAE)',
            'epoch': 'Epoch',
            'train_loss': 'Training Loss',
            'val_loss': 'Validation Loss',
            'train_mae': 'Training MAE',
            'val_mae': 'Validation MAE',
            'true_temp': 'True Temperature',
            'pred_temp': 'Predicted Temperature',
            'sample_index': 'Sample Index',
            'temperature': 'Temperature (°C)',
            'comparison': f'Temperature Prediction\nMSE={mse:.2f}, R²={r2:.2f}',
            'scatter': 'True vs Predicted',
            'true_temp_scatter': 'True Temperature (°C)',
            'pred_temp_scatter': 'Predicted Temperature (°C)',
            'error': 'Prediction Error',
            'error_dist': 'Prediction Error Distribution',
            'error_value': 'Error (°C)',
            'frequency': 'Frequency',
            'error_mean': f'Mean Error: {mae:.2f}°C',
            'residual': 'Residual Plot',
            'pred_temp_residual': 'Predicted Temperature (°C)',
            'residual_value': 'Residual (True-Pred) (°C)'
        }
    
    # 创建2x3的子图
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 训练损失曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history.history['loss'], label=labels['train_loss'], linewidth=2, alpha=0.8)
    ax1.plot(history.history['val_loss'], label=labels['val_loss'], linewidth=2, alpha=0.8)
    ax1.set_xlabel(labels['epoch'])
    ax1.set_ylabel(labels['loss'])
    ax1.set_title('Training History - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history.history['mae'], label=labels['train_mae'], linewidth=2, alpha=0.8)
    ax2.plot(history.history['val_mae'], label=labels['val_mae'], linewidth=2, alpha=0.8)
    ax2.set_xlabel(labels['epoch'])
    ax2.set_ylabel(labels['mae'])
    ax2.set_title('Training History - MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测对比（前100个样本）
    ax3 = plt.subplot(2, 3, 3)
    n_show = min(100, len(y_true))
    ax3.plot(y_true[:n_show], label=labels['true_temp'], linewidth=2, alpha=0.8, color='blue')
    ax3.plot(y_pred[:n_show], label=labels['pred_temp'], linewidth=2, alpha=0.8, color='red')
    ax3.set_xlabel(labels['sample_index'])
    ax3.set_ylabel(labels['temperature'])
    ax3.set_title(labels['comparison'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 真实值 vs 预测值散点图
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(y_true, y_pred, alpha=0.5, s=20, color='green')
    # 添加完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel(labels['true_temp_scatter'])
    ax4.set_ylabel(labels['pred_temp_scatter'])
    ax4.set_title(labels['scatter'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 误差分布直方图
    ax5 = plt.subplot(2, 3, 5)
    errors = y_true - y_pred
    ax5.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel(labels['error_value'])
    ax5.set_ylabel(labels['frequency'])
    ax5.set_title(f"{labels['error_dist']}\n{labels['error_mean']}")
    ax5.grid(True, alpha=0.3)
    
    # 6. 残差图
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(y_pred, errors, alpha=0.5, s=20, color='purple')
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel(labels['pred_temp_residual'])
    ax6.set_ylabel(labels['residual_value'])
    ax6.set_title(labels['residual'])
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'LSTM Model Results - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('../results/lstm_improved_results.png', dpi=150, bbox_inches='tight')
    print("可视化结果已保存: ../results/lstm_improved_results.png")
    plt.show()
    
    # 保存预测结果为CSV
    results_df = pd.DataFrame({
        'True_Temperature': y_true,
        'Predicted_Temperature': y_pred,
        'Error': errors,
        'Absolute_Error': np.abs(errors)
    })
    results_df.to_csv('../results/lstm_predictions.csv', index=False)
    print("预测结果已保存: ../results/lstm_predictions.csv")

# ==================== 主程序 ====================
if __name__ == "__main__":
    try:
        print("开始训练改进版LSTM模型...")
        mse, model = train_and_evaluate_improved()
        
        if mse is not None:
            print(f"\n" + "="*70)
            print("训练完成！")
            print("="*70)
            print(f"最终测试集MSE: {mse:.4f} (°C)²")
            print(f"RMSE: {np.sqrt(mse):.4f} °C")
            print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
            
            # 结果评估
            if mse < 5:
                print("\n结果优秀！模型性能很好。")
            elif mse < 10:
                print("\n结果良好！达到了作业要求。")
            else:
                print("\n结果一般，可以考虑进一步优化。")
                
        else:
            print("\n训练失败，请检查错误信息。")
            
    except Exception as e:
        print(f"\n运行时错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n可能的解决方案:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保有足够的GPU内存")
        print("3. 检查Python包是否安装完整")