import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, csv_path, feature_cols=None, label_col='label'):
        df = pd.read_csv(csv_path)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith('f')]
        self.X = df[feature_cols].values.astype('float32')
        self.y = df[label_col].values.astype('int64')
        self.feature_cols = feature_cols
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])

def make_dataloader(csv_path, batch_size=32, shuffle=True):
    ds = TabularDataset(csv_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), ds
