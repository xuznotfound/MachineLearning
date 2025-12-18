'''
python lstm_delhi_report.py --lookback 14 --batch_size 512 --epochs 200 --num_workers 32 --patience 30 --min_delta 1e-4
'''
import argparse
import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import kagglehub


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int):
        self.X, self.y = self._build_sequences(features, targets, lookback)

    @staticmethod
    def _build_sequences(features: np.ndarray, targets: np.ndarray, lookback: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for i in range(len(features) - lookback):
            xs.append(features[i : i + lookback])
            ys.append(targets[i + lookback])
        X = torch.from_numpy(np.stack(xs)).float()
        y = torch.from_numpy(np.stack(ys)).float()
        return X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(-1)


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df_feat = df.copy()
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat["month"] = df_feat["date"].dt.month.astype(np.float32)
    df_feat["day_of_year"] = df_feat["date"].dt.dayofyear.astype(np.float32)
    df_feat["temp_lag1"] = df_feat["meantemp"].shift(1)
    df_feat["temp_lag7"] = df_feat["meantemp"].shift(7)
    df_feat["temp_rolling_mean_7"] = df_feat["meantemp"].rolling(window=7).mean()
    df_feat["humidity_rolling_mean_7"] = df_feat["humidity"].rolling(window=7).mean()
    df_feat = df_feat.fillna(method="bfill").fillna(method="ffill")
    feature_cols = [
        "humidity",
        "meanpressure",
        "wind_speed",
        "month",
        "day_of_year",
        "temp_lag1",
        "temp_lag7",
        "temp_rolling_mean_7",
        "humidity_rolling_mean_7",
    ]
    return df_feat[feature_cols].astype(np.float32), df_feat["meantemp"].astype(np.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        mse += torch.sum((pred - y) ** 2).item()
        n += y.numel()
    return mse / max(n, 1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def plot_training(history: List[Tuple[int, float, float]], out_dir: Path):
    epochs = [h[0] for h in history]
    train_mse = [h[1] for h in history]
    val_mse = [h[2] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mse, label="Train MSE", linewidth=2)
    plt.plot(epochs, val_mse, label="Test MSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardized)")
    plt.title("Training vs Test MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = out_dir / "training_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # first 100
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:100], label="True", linewidth=2)
    plt.plot(y_pred[:100], label="Pred", linewidth=2)
    plt.title("Prediction Comparison (First 100)")
    plt.xlabel("Sample")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p1 = out_dir / "pred_first100.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, color="tab:green")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    plt.xlabel("True (°C)")
    plt.ylabel("Pred (°C)")
    plt.title("True vs Pred")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p2 = out_dir / "scatter.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # residuals
    errors = y_true - y_pred
    plt.figure(figsize=(10, 4))
    plt.hist(errors, bins=30, edgecolor="black", alpha=0.7, color="tab:orange")
    plt.axvline(x=0, color="r", linestyle="--")
    plt.xlabel("Residual (True - Pred) °C")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)
    p3 = out_dir / "error_hist.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=150)
    plt.close()

    # residual vs pred
    plt.figure(figsize=(10, 4))
    plt.scatter(y_pred, errors, alpha=0.5, s=10, color="tab:purple")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Pred (°C)")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)
    p4 = out_dir / "residual_plot.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=150)
    plt.close()

    return [p1, p2, p3, p4]


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    if args.train_csv is None or args.test_csv is None:
        dataset_root = Path(kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data"))
        print(f"Path to dataset files: {dataset_root}")
        train_csv = dataset_root / "DailyDelhiClimateTrain.csv"
        test_csv = dataset_root / "DailyDelhiClimateTest.csv"
    else:
        train_csv = args.train_csv
        test_csv = args.test_csv

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_features_df, train_target_series = build_features(train_df)
    test_features_df, test_target_series = build_features(test_df)

    train_features = train_features_df.values
    test_features = test_features_df.values
    train_target = train_target_series.values
    test_target = test_target_series.values

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_features = x_scaler.fit_transform(train_features)
    test_features = x_scaler.transform(test_features)
    train_target = y_scaler.fit_transform(train_target.reshape(-1, 1)).squeeze(1)
    test_target = y_scaler.transform(test_target.reshape(-1, 1)).squeeze(1)

    train_ds = SequenceDataset(train_features, train_target, args.lookback)
    test_ds = SequenceDataset(test_features, test_target, args.lookback)

    requested_workers = args.num_workers if args.num_workers is not None else 32
    num_workers = min(requested_workers, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    input_size = train_features.shape[1]
    model = LSTMRegressor(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.MSELoss()

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    best_mse = float("inf")
    best_state = None
    best_epoch = 0
    patience_ctr = 0
    history: List[Tuple[int, float, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(X)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * y.size(0)

        train_mse = epoch_loss / len(train_ds)
        test_mse = evaluate(model, test_loader, device)
        history.append((epoch, train_mse, test_mse))
        print(f"Epoch {epoch:03d} | train MSE {train_mse:.4f} | test MSE {test_mse:.4f}")

        improved = test_mse < best_mse - args.min_delta
        if improved:
            best_mse = test_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best test MSE {best_mse:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model from epoch {best_epoch} with test MSE {best_mse:.4f}")

    # Final metrics in original scale
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            pred = model(X).cpu().numpy()
            preds.append(pred)
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds_denorm = y_scaler.inverse_transform(preds.reshape(-1, 1)).squeeze(1)
    trues_denorm = y_scaler.inverse_transform(trues.reshape(-1, 1)).squeeze(1)

    metrics = compute_metrics(trues_denorm, preds_denorm)
    print("\nFinal metrics (original scale):")
    for k, v in metrics.items():
        if k == "mape":
            print(f"  {k.upper()}: {v:.2f}%")
        else:
            print(f"  {k.upper()}: {v:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_plot = plot_training(history, out_dir)
    pred_plots = plot_predictions(trues_denorm, preds_denorm, out_dir)

    pd.DataFrame({"y_true": trues_denorm, "y_pred": preds_denorm}).to_csv(out_dir / "predictions.csv", index=False)
    print(f"Saved predictions to {out_dir / 'predictions.csv'}")
    print(f"Saved training curve to {train_plot}")
    for p in pred_plots:
        print(f"Saved plot: {p}")

    # Textual summary
    print("\nSummary:")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Test MSE (orig): {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}°C, MAE: {metrics['mae']:.4f}°C, R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Delhi climate with rich reporting")
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--test_csv", type=Path, default=None)
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=Path, default=Path("results/report_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
