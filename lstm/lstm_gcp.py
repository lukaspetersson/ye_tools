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

cuda_availability = torch.cuda.is_available()
if cuda_availability:
  device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
  device = 'cpu'

seq_len = 20
batch_size = 64
lr = 0.001
lstm_layers = 2
lstm_hidden_dim = 20

dataset = DatasetTupelBased('data/tuple_based_big/test', seq_len, device)

model = LstmModel(2, lstm_hidden_dim, lstm_layers).to(device)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
total_loss = train(dataset, model, 2, nn.MSELoss(), opt, batch_size, device=device)

torch.save(model.state_dict(), 'models/gcp/test.pth')
