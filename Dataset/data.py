# 加载数据集并划分训练/测试集
# RevIN-compatible: IQR outliers + raw feature values
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, keep_features=None):
        # 读取数据
        self.data = pd.read_csv(csv_file, encoding='utf-8')
        self.transform = transform

        # ── IQR 异常值处理──────────────────────
        for col in self.data.columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_mask = (self.data[col] < lower) | (self.data[col] > upper)
            n_outliers = outlier_mask.sum()

            if n_outliers > 0:
                clean = self.data[col].copy()
                clean[outlier_mask] = np.nan
                clean = clean.interpolate(method='linear', limit_direction='both')
                clean = clean.bfill().ffill()
                self.data[col] = clean

        # 目标变量
        y_col = 'BTP'

        # 特征筛选（BTP相关前15特征）
        if keep_features is not None:
            valid_features = [f for f in keep_features if f in self.data.columns]
            self.btp_features = self.data[valid_features]
        else:
            corr = self.data.corr()[y_col].abs().sort_values(ascending=False)
            top_n = 15
            valid_features = corr[1:top_n + 1].index.tolist()
            self.btp_features = self.data[valid_features]

        # 热工特征定义（thermal_features）
        self.thermal_features_def = {
            'gas_mass_flow': ['二号风机风量'],
            'T_g_winbox': ['16风箱废气温度', '22风箱废气温度'],
            'solid_fuel': ['焦粉', '烧结用白煤'],
            'bed_thickness': ['料层厚度'],
            'air_flow': ['助燃风流量'],
            'gas_flow': ['煤气流量']
        }
        thermal_cols = []
        for sublist in self.thermal_features_def.values():
            if sublist is not None:
                thermal_cols.extend(sublist)
        thermal_cols = list(dict.fromkeys(thermal_cols))
        self.thermal_features = self.data[thermal_cols]

        # 合并输入特征（BTP相关特征 + 热工特征）
        self.X_combined = pd.concat(
            [self.btp_features, self.thermal_features], axis=1
        )
        self.input_features = self.X_combined.columns.tolist()

        # ── 输入特征：保留原始物理值（RevIN 做 per-sample norm）───
        self.X_raw = self.X_combined.values.astype(np.float32)

        # ── 热工特征：StandardScaler（用于 loss 计算）──────────────
        self.scaler_thermal = StandardScaler()
        self.thermal_features_scaled = self.scaler_thermal.fit_transform(
            self.thermal_features
        )

        # ── BTP 目标变量：StandardScaler（用于 loss & 反归一化）───
        self.Y = self.data[y_col]
        self.y_scaler = StandardScaler()
        self.Y_scaled = pd.Series(
            self.y_scaler.fit_transform(
                self.Y.values.reshape(-1, 1)
            ).flatten(),
            name=y_col
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 输入：原始物理值（RevIN 在模型 forward 中做 norm）
        X_raw = self.X_raw[idx]
        # BTP 目标：StandardScaler 标准化
        Y = self.Y_scaled.iloc[idx]
        # 热工特征：StandardScaler 标准化（用于计算 loss）
        thermal_original = self.thermal_features_scaled[idx]

        sample = {
            'X': torch.tensor(X_raw, dtype=torch.float32),
            'Y': torch.tensor(Y, dtype=torch.float32),
            'thermal_true': torch.tensor(thermal_original, dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_data():
    csv_file = 'data.csv'
    keep_features = None
    custom_dataset = CustomDataset(csv_file, keep_features=keep_features)

    # 训练/测试集划分：80/20
    train_size = int(0.8 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = random_split(
        custom_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for batch in train_loader:
        print(f"合并输入特征形状: {batch['X'].shape}")
        print(f"BTP目标形状: {batch['Y'].shape}")
        print(f"热工特征形状: {batch['thermal_true'].shape}")
        print(f"使用的输入特征: {custom_dataset.input_features}")
        break

    return train_loader, test_loader, custom_dataset
