import os

import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

class synthetic_dataset(Dataset):

    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def __getitem__(self, idx):
        
        X = self.data[idx]
        y = self.labels(idx)

        return X, y

    def __len__(self):
        
        l = len(self.data)

        return l