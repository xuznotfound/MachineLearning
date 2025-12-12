# 进入LSTM目录
cd ~/homework/lstm_climate

# 备份原来的代码
cp lstm_climate_temp.py lstm_climate_temp.py.backup

# 创建新的适配代码
cat > lstm_climate_final.py << 'EOF'
#!/usr/bin/env python3
"""
LSTM模型用于德里气候数据温度预测
使用湿度、气压、风速三个指标预测温度
适配您下载的数据集结构（有独立的训练集和测试集）
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置GPU配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU可用: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU设置错误: {e}")

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("LSTM模型 - 德里气候温度预测")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==================== 1. 加载数据集 ====================
def load_climate_datasets():
    """加载训练集和测试集"""
    print("\n[1/6] 正在加载气候数据集...")
    
    train_path = '../data/DailyDelhiClimateTrain.csv'
    test_path = '../data/DailyDelhiClimateTest.csv'
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        print(f"? 训练集文件不存在: {train_path}")
        return None, None
    
    if not os.path.exists(test_path):
        print(f"? 测试集文件不存在: {test_path}")
        return None, None
    
    try:
        # 加载训练集
        print(f"  加载训练集: {train_path}")
        df_train = pd.read_csv(train_path)
        print(f"  训练集形状: {df_train.shape}")
        
        # 加载测试集
        print(f"  加载测试集: {test_path}")
        df_test = pd.read_csv(test_path)
        print(f"  测试集形状: {df_test.shape}")
        
        # 合并数据集（用于统一预处理）
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        print(f"  合并后总数据形状: {df_combined.shape}")
        
        return df_train, df_test, df_combined
        
    except Exception as e:
        print(f"? 加载数据失败: {e}")
        return None, None, None

# ==================== 2. 数据预处理 ====================
def preprocess_data(df):
    """数据预处理"""
    print("\n[2/6] 数据预处理...")
    
    # 复制数据，避免修改原始数据
    df_processed = df.copy()
    
    # 检查必要的列
    required_columns = ['date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure']
    missing_cols = [col for col in required_columns if col not in df_processed.columns]
    
    if missing_cols:
        print(f"? 缺少必要列: {missing_cols}")
        print(f"  可用列: {df_processed.columns.tolist()}")
        return None
    
    # 处理日期
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed = df_processed.sort_values('date').reset_index(drop=True)
    
    # 检查缺失值
    print("  缺失值统计:")
    print(df_processed[required_columns].isnull().sum())
    
    # 填充缺失值
    df_processed[required_columns[1:]] = df_processed[required_columns[1:]].fillna(method='ffill')
    
    # 数据统计
    print("\n  数据统计摘要:")
    print(df_processed[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].describe())
    
    return df_processed

# ==================== 3. 创建时间序列数据集 ====================
def create_sequences_from_data(df, window_size=7):
    """从数据创建时间序列数据集"""
    
    # 选择特征：湿度、气压、风速
    features = ['humidity', 'meanpressure', 'wind_speed']
    target = 'meantemp'
    
    # 提取数据
    X = df[features].values
    y = df[target].values
    
    # 创建序列
    X_seq, y_seq = [], []
    for i in range(window_size, len(df)):
        X_seq.append(X[i-window_size:i])  # 过去window_size天的特征
        y_seq.append(y[i])  # 当前天的温度
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"  创建序列: X={X_seq.shape}, y={y_seq.shape}")
    print(f"  每个样本包含: {window_size} 天 × {len(features)} 个特征")
    
    return X_seq, y_seq, features, target

# ==================== 4. 构建LSTM模型 ====================
def build_lstm_model(input_shape):
    """构建LSTM模型用于温度预测"""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 输出温度值
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # 均方误差
        metrics=['mae', 'mse']  # 平均绝对误差和均方误差
    )
    
    return model

# ==================== 5. 训练和评估LSTM ====================
def train_and_evaluate():
    """训练并评估LSTM模型"""
    
    # 加载数据
    df_train, df_test, df_combined = load_climate_datasets()
    if df_train is None or df_test is None:
        return
    
    # 预处理训练集和测试集
    print("\n预处理训练集...")
    df_train_processed = preprocess_data(df_train)
    
    print("\n预处理测试集...")
    df_test_processed = preprocess_data(df_test)
    
    if df_train_processed is None or df_test_processed is None:
        print("? 数据预处理失败")
        return
    
    # 创建序列数据
    window_size = 7  # 使用过去7天的数据
    print(f"\n[3/6] 创建时间序列数据集 (窗口大小={window_size})...")
    
    # 从训练集创建序列
    X_train_seq, y_train_seq, features, target = create_sequences_from_data(
        df_train_processed, window_size
    )
    
    # 从测试集创建序列（注意：测试集需要独立创建，不能混合）
    X_test_seq, y_test_seq, _, _ = create_sequences_from_data(
        df_test_processed, window_size
    )
    
    print(f"\n  训练序列: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}")
    print(f"  测试序列: X_test={X_test_seq.shape}, y_test={y_test_seq.shape}")
    
    # 数据归一化
    print("\n[4/6] 数据归一化...")
    
    # 注意：对于时间序列，应该只使用训练集的统计信息来归一化测试集
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 重塑训练数据进行归一化
    X_train_reshaped = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    X_test_reshaped = X_test_seq.reshape(-1, X_test_seq.shape[-1])
    
    # 只在训练集上拟合归一化器
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train_seq.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test_seq.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_seq.reshape(-1, 1)).flatten()
    
    # 构建模型
    print("\n[5/6] 构建LSTM模型...")
    input_shape = (window_size, len(features))
    model = build_lstm_model(input_shape)
    model.summary()
    
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # 设置回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
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
            '../models/best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print("\n[6/6] 开始训练LSTM模型...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # ==================== 6. 评估模型 ====================
    print("\n评估模型...")
    
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
    
    print(f"\n? 测试集评估结果:")
    print(f"  MSE (均方误差): {mse:.4f} (°C)?")
    print(f"  RMSE (均方根误差): {rmse:.4f} °C")
    print(f"  MAE (平均绝对误差): {mae:.4f} °C")
    print(f"  R? 分数: {r2:.4f}")
    
    # ==================== 7. 可视化结果 ====================
    # 训练历史
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失', linewidth=2)
    plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE损失', fontsize=12)
    plt.title('LSTM训练历史 - 损失', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='验证MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE (°C)', fontsize=12)
    plt.title('LSTM训练历史 - MAE', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/lstm_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 预测结果对比
    plt.figure(figsize=(14, 10))
    
    # 整体预测
    plt.subplot(3, 2, 1)
    plt.plot(y_test_original, label='真实温度', linewidth=2, alpha=0.8, color='blue')
    plt.plot(y_pred, label='预测温度', linewidth=2, alpha=0.8, color='red')
    plt.xlabel('测试集样本索引', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.title('LSTM温度预测结果 (全部测试集)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 前50个样本详细对比
    plt.subplot(3, 2, 2)
    n_show = min(50, len(y_test_original))
    plt.plot(y_test_original[:n_show], label='真实温度', linewidth=2, alpha=0.8, color='blue')
    plt.plot(y_pred[:n_show], label='预测温度', linewidth=2, alpha=0.8, color='red')
    plt.xlabel('测试集样本索引', fontsize=12)
    plt.ylabel('温度 (°C)', fontsize=12)
    plt.title(f'LSTM温度预测结果 (前{n_show}个样本)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 散点图
    plt.subplot(3, 2, 3)
    plt.scatter(y_test_original, y_pred, alpha=0.5, s=20, color='green')
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 
             'r--', linewidth=2, label='完美预测')
    plt.xlabel('真实温度 (°C)', fontsize=12)
    plt.ylabel('预测温度 (°C)', fontsize=12)
    plt.title('真实值 vs 预测值', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 误差分布
    plt.subplot(3, 2, 4)
    errors = y_test_original - y_pred
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('预测误差 (°C)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('预测误差分布', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 残差图
    plt.subplot(3, 2, 5)
    plt.scatter(y_pred, errors, alpha=0.5, s=20, color='purple')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('预测温度 (°C)', fontsize=12)
    plt.ylabel('残差 (真实 - 预测)', fontsize=12)
    plt.title('预测残差图', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 误差随时间变化
    plt.subplot(3, 2, 6)
    plt.plot(errors, alpha=0.7, color='brown')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.fill_between(range(len(errors)), errors, 0, where=(errors >= 0), 
                     color='red', alpha=0.3, label='高估')
    plt.fill_between(range(len(errors)), errors, 0, where=(errors < 0), 
                     color='blue', alpha=0.3, label='低估')
    plt.xlabel('测试集样本索引', fontsize=12)
    plt.ylabel('预测误差 (°C)', fontsize=12)
    plt.title('预测误差随时间变化', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/lstm_predictions_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存模型
    model.save('../models/lstm_final_model.h5')
    
    print(f"\n{'='*70}")
    print(f"LSTM模型训练完成!")
    print(f"测试集MSE: {mse:.4f} (°C)?")
    print(f"模型已保存到: ../models/lstm_final_model.h5")
    print(f"可视化结果已保存到: ../results/")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    return mse

# ==================== 主程序 ====================
if __name__ == "__main__":
    try:
        # 训练和评估模型
        mse = train_and_evaluate()
        if mse is not None:
            print(f"\n? LSTM作业完成！测试集MSE: {mse:.4f} (°C)?")
        else:
            print("\n? LSTM训练失败")
    except Exception as e:
        print(f"\n? 运行出错: {e}")
        import traceback
        traceback.print_exc()
EOF