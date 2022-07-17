##
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn.functional import relu
import matplotlib.pyplot as plt
import music21 as m21

torch.manual_seed(1)

SEQ_LEN = 5

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
        x = torch.view_as_complex(torch.tensor(part[idx:idx+self.seq_len], dtype=torch.float))
        y = torch.view_as_complex(torch.tensor([part[idx+self.seq_len]], dtype=torch.float))
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
    loss_vals = []
    num_epochs = 5

    dataloader = DataLoader(dataset, batch_size=64)
    loss_fn = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in dataloader:
            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        loss_vals.append(epoch_loss)
        print(epoch_loss)
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals)


##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CvnnModel(layer_dims=[SEQ_LEN, 10, 4]).to(device)
dataset = Dataset("data/data_small.npy", SEQ_LEN)
train(dataset, model)

##
model.eval()
x = torch.view_as_complex(torch.tensor([[45,2],[45,2],[48,4], [43,2], [45,2]], dtype=torch.float))
model(x)

##
def generate(model, start, song_len):
    model.eval()
    start_len = len(start)
    seq = torch.cat((start, torch.zeros(song_len-start_len, dtype=torch.cfloat)))
    for i in range(start_len, song_len):
        x = seq[i-SEQ_LEN:i]
        y = model(x)
        seq[i] = y
    return seq

def to_stream(seq):
    stream = m21.stream.Stream()
    for cn in seq:
        note = m21.note.Note(pitch=round(cn.real.item()), duration=m21.duration.Duration(round(cn.imag.item())/4))
        stream.append(note)
    return stream

##
seq = generate(model, x, 30)
stream = to_stream(seq)
stream.show()

##
mf = m21.midi.translate.streamToMidiFile(stream)
mf.open('./data/test_output.midi', 'wb')
mf.write()
mf.close()







