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
from lstm import DatasetTupelBased, DatasetEventBased, train, LstmModel

torch.manual_seed(1)
device = "cuda"

SEQ_LENS = [30]
PARAMETERS = dict(
    batch_size = [1024],
    lr = [0.01],
    lstm_layers = [1],
    lstm_hidden_dim = [20],
    optimizer = ['sgd', 'adam'],
)

model_id = 0
for seq_len in SEQ_LENS:
    print("seq len", seq_len)
    dataset = DatasetTupelBased(seq_len)
    print("len dataset: ", len(dataset))
    for batch_size, lr, lstm_layers, lstm_hidden_dim, optimizer, momentum ,dropout in product(*[v for v in PARAMETERS.values()]): 
        comment = f' seq_len = {seq_len}, batch_size = {batch_size}, lr = {lr}, layers = {lstm_layers}, hidden_dim = {lstm_hidden_dim}, optimizer = {optimizer}, dropout = {dropout}, model_id = {model_id}'
        print(comment)
        writer = SummaryWriter('runs/sun5', comment=comment)
        model = LstmModel(2, lstm_hidden_dim, lstm_layers, dropout).to(device)
        if optimizer == 'adam':
            opt = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        t0 = time.time()
        total_loss = train(dataset, model, 4, nn.MSELoss(), opt, batch_size, writer)
        train_time = time.time()-t0
        writer.add_hparams(dict(seq_len=seq_len, batch_size=batch_size, lr=lr, lstm_layers=lstm_layers, lstm_hidden_dim=lstm_hidden_dim, optimizer=optimizer, dropout=dropout), dict(loss=total_loss, train_time=train_time))
        writer.flush()
        writer.close()
        torch.save(model.state_dict(), 'models/hyperparameter/note_based_'+str(model_id)+'_'+time.strftime("%m-%d")+'.pth')
        print("model saved")
        model_id += 1
