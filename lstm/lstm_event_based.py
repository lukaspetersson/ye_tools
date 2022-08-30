import time
import encode_midi as event_utils
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

torch.manual_seed(1)
device = "cpu"

SEQ_LENS = [20, 100]
PARAMETERS = dict(
    batch_size = [1024],
    lr = [0.01, 0.001],
    lstm_layers = [1, 2],
    lstm_hidden_dim = [10, 30],
    optimizer = ['sgd', 'adam'],
    dropout = [0.2, 0]
)

class DatasetEventBased(torch.utils.data.Dataset):
    def __init__(self, file_paths, seq_len):
        def f(fp):
            try:
                return list(filter(lambda x: x<356, event_utils.encode_midi(fp)))
            except:
                return []

        self.data = [f(file_path) for file_path in file_paths] 
        self.seq_len = seq_len 

    def data_in_part(self, part):
        return len(part) - (self.seq_len + 1)

    def __len__(self):
        l = 0
        for part in self.data:
            l += self.data_in_part(part)
        return l

    def get_part(self, idx):
        i = idx
        for song in self.data:
            if i - self.data_in_song(song) < 0:
                break
            else:
                i -= self.data_in_song(song)
        return (song, i)

    def __getitem__(self, idx):
        for part in self.data:
            num_datapoints = self.data_in_part(part)
            if idx - num_datapoints < 0:
                break
            else:
                idx -= num_datapoints
        x = F.one_hot(torch.tensor(part[idx:idx+self.seq_len]), num_classes=356).float().to(device)
        y = F.one_hot(torch.tensor(part[idx+1:idx+self.seq_len+1]), num_classes=356).float().to(device)
        return (x, y)

class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LstmModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout)
        self.linear = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state

def train(dataset, model, num_epochs, loss_fn, opt, batch_size, writer):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0
    for epoch in range(num_epochs):
        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim).to(device)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim).to(device)
        for x, y in dataloader:
            opt.zero_grad()
            y_pred, (h_t, c_t) = model(x, (h_t, c_t))
            loss = loss_fn(y_pred, y)

            writer.add_scalar("Loss", loss, epoch)
            total_loss += loss.item()

            h_t = h_t.detach()
            c_t = c_t.detach()

            loss.backward()
            opt.step()
    return total_loss

model_id = 0
for seq_len in SEQ_LENS:
    print("seq len", seq_len)
    dataset = DatasetEventBased(file_paths=list(glob.glob('data/lmd_full/**/*.mid', recursive=True))[:5], seq_len=seq_len)
    print("len dataset: ", len(dataset))
    for batch_size, lr, lstm_layers, lstm_hidden_dim, optimizer, dropout in product(*[v for v in PARAMETERS.values()]): 
        comment = f' seq_len = {seq_len}, batch_size = {batch_size}, lr = {lr}, layers = {lstm_layers}, hidden_dim = {lstm_hidden_dim}, optimizer = {optimizer}, dropout = {dropout}, model_id = {model_id}'
        print(comment)
        writer = SummaryWriter('runs/event1', comment=comment)
        model = LstmModel(356, lstm_hidden_dim, lstm_layers, dropout).to(device)
        if optimizer == 'adam':
            opt = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
        t0 = time.time()
        total_loss = train(dataset, model, 2, nn.CrossEntropyLoss(), opt, batch_size, writer)
        train_time = time.time()-t0
        writer.add_hparams(dict(seq_len=seq_len, batch_size=batch_size, lr=lr, lstm_layers=lstm_layers, lstm_hidden_dim=lstm_hidden_dim, optimizer=optimizer, dropout=dropout), dict(loss=total_loss, train_time=train_time))
        writer.flush()
        writer.close()
        torch.save(model.state_dict(), 'models/event_based_'+str(model_id)+'_'+time.strftime("%m-%d")+'.pth')
        print("model saved")
        model_id += 1