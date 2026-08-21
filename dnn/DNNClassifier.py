import numpy as np
import torch
import torch.nn as nn  # Нейронные слои (Linear, Conv2d, LSTM и т.д.)
import torch.optim as optim  # Оптимизаторы (Adam, SGD)
import torch.nn.functional as F  # Функции активации, свертки, пулинг (без состояния)
from torch.utils.data import DataLoader, Dataset  # Для загрузки данных


class DNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim):
        super(DNNClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(in_features=input_dim, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.layers(x)
