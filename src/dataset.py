"""Dataset for the Kaggle MIT-BIH heartbeat CSV (Kachuee et al.).

Each row: 187 signal samples + 1 integer class label (0..4).
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGBeatDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        self.X = df.iloc[:, :187].values.astype("float32")
        self.y = df.iloc[:, 187].values.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).unsqueeze(0)  # shape [1, 187] (1 channel)
        return x, int(self.y[i])
