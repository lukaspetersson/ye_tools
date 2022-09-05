import time
import tempfile
import numpy as np
import glob
import tqdm
import torch
from torch import nn, optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import music21 as m21
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from utils import DatasetTupelBased, DatasetEventBased, train, LstmModel

torch.manual_seed(1)
device = "cpu"

seq_len = 20
batch_size = 128
lr = 0.005
lstm_layers = 1
lstm_hidden_dim = 20

dataset = DatasetTupelBased(seq_len)

model = LstmModel(2, lstm_hidden_dim, lstm_layers).to(device)

opt = optim.Adam(model.parameters(), lr=lr)
total_loss = train(dataset, model, 6, nn.MSELoss(), opt, batch_size)

torch.save(model.state_dict(), 'models/gcp/test.pth')
