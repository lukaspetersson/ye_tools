##
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.functional import relu

torch.manual_seed(1)

SEQ_LEN = 10

##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len=SEQ_LEN):
        self.data = np.load(data_dir, allow_pickle=True)
        self.seq_len = seq_len

    def data_in_part(self, part):
        return len(part) - (self.seq_len + 1)

    def __len__(self):
        l = 0
        for part in self.data:
            l += self.data_in_song(part)
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
        x = torch.view_as_complex(torch.tensor(part[i:i+self.seq_len], dtype=torch.float))
        y = torch.view_as_complex(torch.tensor(part[i+self.seq_len], dtype=torch.float))
        return (x, y)

##
def complex_relu(z):
    return relu(z.real) + 1.j * relu(z.imag)

class CvnnModel(nn.Module):
    def __init__(self, layer_dims):
        super(CvnnModel, self).__init__()
        self.l1 = nn.Linear(layer_dims[0], layer_dims[1]).to(torch.cfloat)
        self.l2 = nn.Linear(layer_dims[1], layer_dims[2]).to(torch.cfloat)
        self.l3 = nn.Linear(layer_dims[2], 1).to(torch.cfloat)
        self.relu = complex_relu

    def forward(self, inputs):
        x = self.l1(inputs)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

##
def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=64)
    loss_fn = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        for x, y in dataloader:
            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()

##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CvnnModel(layer_dims=[SEQ_LEN, 20, 5]).to(device)
dataset = Dataset("data/data_small.npy", SEQ_LEN)
train(dataset, model)

##
#TODO: bug in dataset?
dataloader = DataLoader(dataset)
dataiter = iter(dataloader)
images, labels = dataiter.next()
print(images)

