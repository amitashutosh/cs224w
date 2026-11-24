"""
price_loader.py

Pipeline for Step 1:
1. Direct ingestion of S&P 500 daily price history (sp500.csv).
2. Per-stock Z-score normalization.
3. 64-day rolling window chunking for temporal forecasting.
4. Global Pearson correlation graph prior calculation (A_global).
5. Chronological 80/10/10 train/val/test PyTorch DataLoaders.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Remote URL for baseline sp500.csv from AlienKevin/sp500
REMOTE_DATA_URL = (
    "https://raw.githubusercontent.com/AlienKevin/sp500/main/sp500.csv"
)
LOCAL_CSV_PATH = "data/sp500.csv"


# =====================================================================
# 1. Dataset & Custom PyTorch Tensor Wrapper
# =====================================================================
class StockPriceDataset(Dataset):
    """
    PyTorch Dataset yielding:
      - x: Input price sequences of shape (window_size, num_stocks)
      - y: Next-day target prices of shape (num_stocks,)
      - dates: Target timestamps corresponding to prediction target y
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, dates: np.ndarray):
        self.x = x
        self.y = y
        self.dates = dates

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        return self.x[idx], self.y[idx], self.dates[idx]


# =====================================================================
# 2. Main Preprocessing & Loader Pipeline
# =====================================================================
class SP500PricePipeline:

    def __init__(
        self,
        csv_path: str = LOCAL_CSV_PATH,
        window_size: int = 64,
        batch_size: int = 32,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ):
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.tickers: List[str] = []
        self.dates: pd.DatetimeIndex = None
        self.means: np.ndarray = None
        self.stds: np.ndarray = None
        self.A_global: torch.Tensor = None

    def load_raw_data(self) -> pd.DataFrame:
        """Loads sp500.csv from local storage, or pulls directly from remote repository."""
        if os.path.exists(self.csv_path):
            print(f"Loading local S&P 500 data from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
        else:
            print(
                f"Local file not found. Ingesting directly from: {REMOTE_DATA_URL}"
            )
            df = pd.read_csv(REMOTE_DATA_URL)
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            df.to_csv(self.csv_path, index=False)
            print(f"Saved local cache to: {self.csv_path}")

        # Set Date index matching the reference Colab implementation
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

        self.tickers = list(df.columns)
        self.dates = df.index
        return df

    def compute_global_correlation(
        self, train_prices: np.ndarray
    ) -> torch.Tensor:
        """
        Computes the static global Pearson correlation matrix across stocks (N x N)
        using only training period data to avoid future correlation leakage.
        """
        # train_prices shape: (T_train, num_stocks)
        # corrcoef expects variables as rows -> shape: (num_stocks, T_train)
        corr_matrix = np.corrcoef(train_prices.T)

        # Handle any zero-variance edge cases and enforce self-loops
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        self.A_global = torch.tensor(corr_matrix, dtype=torch.float32)
        return self.A_global

    def _create_chunks(
        self, data: np.ndarray, dates: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Generates contiguous rolling chunks of length T=64 and target t+1.
        x: (num_samples, 64, num_stocks)
        y: (num_samples, num_stocks)
        """
        x_list, y_list, target_dates = [], [], []
        num_timesteps = len(data)

        for i in range(num_timesteps - self.window_size):
            x_list.append(data[i : i + self.window_size])
            y_list.append(data[i + self.window_size])
            target_dates.append(str(dates[i + self.window_size].date()))

        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
        return x_tensor, y_tensor, np.array(target_dates)

    def build_loaders(self) -> Dict[str, DataLoader]:
        """
        Executes end-to-end normalization, chronological partitioning,
        and DataLoader generation.
        """
        df = self.load_raw_data()
        raw_values = df.values  # Shape: (T, N)
        num_timesteps, num_stocks = raw_values.shape

        print(
            f"Dataset Ingested: {num_timesteps} trading days across {num_stocks} stocks."
        )
        print(f"Date Coverage: {self.dates[0].date()} to {self.dates[-1].date()}")

        # Strict chronological 80 / 10 / 10 split indices
        train_end_idx = int(num_timesteps * self.train_ratio)
        val_end_idx = int(num_timesteps * (self.train_ratio + self.val_ratio))

        train_raw = raw_values[:train_end_idx]
        val_raw = raw_values[train_end_idx:val_end_idx]
        test_raw = raw_values[val_end_idx:]

        train_dates = self.dates[:train_end_idx]
        val_dates = self.dates[train_end_idx:val_end_idx]
        test_dates = self.dates[val_end_idx:]

        # Compute static Global Pearson Correlation Matrix on train slice
        self.compute_global_correlation(train_raw)

        # Per-stock Z-score Normalization (fitted strictly on training set)
        self.means = np.mean(train_raw, axis=0, keepdims=True)
        self.stds = np.std(train_raw, axis=0, keepdims=True)
        self.stds[self.stds == 0.0] = 1.0  # Prevent divide-by-zero

        train_norm = (train_raw - self.means) / self.stds
        val_norm = (val_raw - self.means) / self.stds
        test_norm = (test_raw - self.means) / self.stds

        # Create 64-day temporal sequences
        x_train, y_train, d_train = self._create_chunks(train_norm, train_dates)
        x_val, y_val, d_val = self._create_chunks(val_norm, val_dates)
        x_test, y_test, d_test = self._create_chunks(test_norm, test_dates)

        # Wrap in PyTorch Datasets
        train_dataset = StockPriceDataset(x_train, y_train, d_train)
        val_dataset = StockPriceDataset(x_val, y_val, d_val)
        test_dataset = StockPriceDataset(x_test, y_test, d_test)

        # Build DataLoaders (Shuffle=False to strictly preserve sequence order)
        loaders = {
            "train": DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False
            ),
            "val": DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            ),
            "test": DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            ),
        }

        print("\n--- Chronological Split & Sequence Verification ---")
        print(
            f"Train Sequences: X={x_train.shape}, Y={y_train.shape} (Span: {train_dates[0].date()} -> {train_dates[-1].date()})"
        )
        print(
            f"Val   Sequences: X={x_val.shape}, Y={y_val.shape} (Span: {val_dates[0].date()} -> {val_dates[-1].date()})"
        )
        print(
            f"Test  Sequences: X={x_test.shape}, Y={y_test.shape} (Span: {test_dates[0].date()} -> {test_dates[-1].date()})"
        )
        print(f"Global Correlation Prior A_global: {self.A_global.shape}")

        return loaders


# =====================================================================
# 3. Sanity Check Execution Block
# =====================================================================
if __name__ == "__main__":
    pipeline = SP500PricePipeline(
        csv_path="data/sp500.csv", window_size=64, batch_size=32
    )

    data_loaders = pipeline.build_loaders()

    # Verify a single batch execution
    sample_x, sample_y, sample_dates = next(iter(data_loaders["train"]))

    print("\n--- Single DataLoader Batch Inspection ---")
    print(f"Batch X Shape (Batch, Lookback, Stocks) : {sample_x.shape}")
    print(f"Batch Y Shape (Batch, Stocks)           : {sample_y.shape}")
    print(f"Number of Target Dates in Batch         : {len(sample_dates)}")
    print(f"First Target Date in Batch              : {sample_dates[0]}")
    print(
        f"Contains NaNs in X                      : {torch.isnan(sample_x).any().item()}"
    )
    print(
        f"Contains NaNs in Y                      : {torch.isnan(sample_y).any().item()}"
    )
    print(
        f"Global Prior Range                      : [{pipeline.A_global.min():.3f}, {pipeline.A_global.max():.3f}]"
    )
    print(
        f"First 5 Constituent Tickers             : {pipeline.tickers[:5]}\n"
    )