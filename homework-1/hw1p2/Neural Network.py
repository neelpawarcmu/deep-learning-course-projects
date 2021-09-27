#Kaggle Neural Network 
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X
        self.length = len(X)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return X[i]

    def collate_fn(batch):
        batch_x = torch.as_tensor(batch)
        return batch_x

class NeuralNetwork:
    def __init__(self):
        pass