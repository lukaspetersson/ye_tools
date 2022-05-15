##
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(1)

##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len):
        self.data = np.load(data_dir, allow_pickle=True)
        self.seq_len = seq_len

    def data_in_song(self, song):
        return len(song) - self.seq_len

    def __len__(self):
        l = 0
        for song in self.data:
            l += self.data_in_song(song)
        return l

    def get_song(self, idx):
        i = idx
        for song in self.data:
            if i - self.data_in_song(song) < 0:
                break
            else:
                i -= self.data_in_song(song)
        return (song, i)

    def __getitem__(self, idx):
        song, i = self.get_song(idx)
        #TODO: what if seq_len > len(song)
        x = torch.tensor(song[i:i+self.seq_len])
        y = torch.tensor(song[i+1:i+self.seq_len+1])
        return (x, y)

##
dataset = Dataset("data/data.npy", 20)
dataset.__getitem__(100)[1].size()

##
class LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(128, self.hidden_dim, num_layers=self.num_layers, dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, 128)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state
##
model = LSTM()

##
def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=64)
    loss_fn = nn.CrossEntropyLoss()
    #TODO: quasi newton?
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (h_t, c_t) = model(x, (h_t, c_t))
            #TODO: dimensions
            loss = loss_fn(y_pred, y)

            h_t = h_t.detach()
            c_t = c_t.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
##
train(dataset, model)
