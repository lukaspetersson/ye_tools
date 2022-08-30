##
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
import encode_midi as event_utils
import music21 as m21
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1)
device = "cuda"
SEQ_LEN = 25
writer = SummaryWriter()

##
# dataset as used by magenta, with veolcity removed
class DatasetEventBased(torch.utils.data.Dataset):
    def __init__(self, file_paths, seq_len):
        def encode_one_song(fp):
            try:
                return list(filter(lambda x: x<356, event_utils.encode_midi(fp)))
            except:
                return []

        self.data = [encode_one_song(file_path) for file_path in file_paths] 
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

##
# Dataset as described in post 3
class DatasetTupelBased(torch.utils.data.Dataset):
    def __init__(self, seq_len=SEQ_LEN):
        self.data = np.load('data/data_big_100.npy', allow_pickle=True)
        #self.data = np.array(np.concatenate([np.load('data/data_big_'+str(i*100)+'.npy', allow_pickle=True) for i in range(1,50)]))

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
        x = torch.tensor(part[idx:idx+self.seq_len], dtype=torch.float32).to(device)
        y = torch.tensor(part[idx+1:idx+self.seq_len+1], dtype=torch.float32).to(device)
        return (x, y)

##
class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=1):
        super(LstmModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state
##
def train(dataset, model, num_epochs, loss_fn):
    model.train()
    loss_vals = []

    dataloader = DataLoader(dataset, batch_size=128)
    #opt = optim.Adam(model.parameters(), lr=0.005)
    opt = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(num_epochs):
        epoch_loss = []

        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim).to(device)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim).to(device)
        for x, y in dataloader:
            opt.zero_grad()
            y_pred, (h_t, c_t) = model(x, (h_t, c_t))
            loss = loss_fn(y_pred, y)

            h_t = h_t.detach()
            c_t = c_t.detach()

            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals)
    plt.title("Loss")
    plt.ticklabel_format(useOffset=False)

##
# Train and save an event based model
model_event = LstmModel(input_dim=356, hidden_dim=30, num_layers=2).to(device)
dataset_event = DatasetEventBased(file_paths=list(glob.glob('data/lmd_full/**/*.mid', recursive=True))[:2], seq_len=SEQ_LEN)
train(dataset_event, model_event, 2, nn.CrossEntropyLoss())
torch.save(model_event.state_dict(), 'models/event_small.pth')

##
# Train and save a tuple based model
model_tup = LstmModel(input_dim=2).to(device)
dataset_tup = DatasetTupelBased(SEQ_LEN)
train(dataset_tup, model_tup, 3, nn.MSELoss())
torch.save(model_tup.state_dict(), 'models/tup_small.pth')

##
# Test a forward pass of event based model
model_event.eval()
h_t = torch.zeros(model_event.num_layers, dataset_event.seq_len, model_event.hidden_dim).to(device)
c_t = torch.zeros(model_event.num_layers, dataset_event.seq_len, model_event.hidden_dim).to(device)
x = torch.unsqueeze(dataset_event[500][0].to(device), dim=0)
y_pred, state = model_event(x, (h_t, c_t))
# write model to tensorboard 
writer.add_graph(model_event, (x, (h_t, c_t)))
torch.argmax(y_pred[0][-1])

##
# Test a forward pass of tuple based model
model_tup.eval()
h_t = torch.zeros(model_tup.num_layers, dataset_tup.seq_len, model_tup.hidden_dim).to(device)
c_t = torch.zeros(model_tup.num_layers, dataset_tup.seq_len, model_tup.hidden_dim).to(device)
x = torch.unsqueeze(dataset_tup[100][0].to(device), dim=0)
y_pred, state = model_tup(x, (h_t, c_t))
# write model to tensorboard 
writer.add_graph(model_tup, (x, (h_t, c_t)))
y_pred[0][-1]

##
# Use model to generate sequance from a starting sequance
def generate(model, start, song_len, seq_len):
    model.eval()
    start_len = len(start)
    seq = torch.cat((start, torch.zeros(song_len-start_len, model.input_dim, dtype=torch.float))).to(device)
    h_t = torch.zeros(model.num_layers, seq_len, model.hidden_dim).to(device)
    c_t = torch.zeros(model.num_layers, seq_len, model.hidden_dim).to(device)
    for i in range(start_len, song_len):
        x = torch.unsqueeze(seq[i-seq_len:i], dim=0)
        y, (h_t, c_t) = model(x, (h_t, c_t))
        seq[i] = y[0][-1]
    return seq

##
# Convert tuple based sequance to m21 stream
def to_stream_tup_based(seq):
    stream = m21.stream.Stream()
    for note in seq:
        note_m21 = m21.note.Note(pitch=round(note[0].item()), duration=m21.duration.Duration(round(note[1].item()/4)))
        stream.append(note_m21)
    return stream

##
start = torch.tensor(np.load('data/twinkle_note_based.npy', allow_pickle=True)[0][:SEQ_LEN], dtype=torch.float)
seq = generate(model_tup, start, len(start)*2, SEQ_LEN)
stream = to_stream_tup_based(seq)
stream.show()

##

def to_stream_event_based(seq):
    seq = [torch.argmax(oh).item() for oh in seq]
    tmp_mid = tempfile.NamedTemporaryFile(suffix='.mid')
    tmp_mid = 'twinkle_event.mid'
    event_utils.decode_midi(seq, tmp_mid)
    return m21.converter.parse(tmp_mid)

##
start = [375, 60, 371, 52, 372, 48, 315, 188, 258, 376, 60, 312, 180, 176, 258, 188, 258, 378, 67, 375, 60, 373, 48, 315, 195, 258, 377, 67, 312, 188, 176, 258, 195, 258, 377, 69, 374, 60, 372, 41, 315, 197, 258, 376, 69, 312, 188, 169, 258, 197, 258, 376, 67, 374, 60, 373, 48, 355, 275, 195, 188, 176, 261, 376, 65, 373, 57, 373, 41, 315, 193, 258, 377, 65, 312, 185, 169, 258, 193, 258, 376, 64, 373, 55, 375, 48, 315, 192, 258, 377, 64, 312, 183, 176, 258, 192, 258, 376, 62, 373, 55, 373, 43, 315, 190, 258, 376, 62, 312, 183, 171, 258, 190, 258, 376, 60, 373, 52, 374, 48, 355, 275, 188, 180, 176, 261, 379, 67, 376, 60, 374, 48, 315, 195, 258, 376, 67, 312, 188, 176, 258, 195, 258, 376, 65, 373, 57, 373, 41, 315, 193, 258, 377, 65, 312, 185, 169, 258, 193, 258, 376]
start = list(filter(lambda x: x<356, start))
start = F.one_hot(torch.tensor(start), num_classes=356).float()
seq_event_based = generate(model_event, start, 2*len(start), SEQ_LEN)
stream = to_stream_event_based(seq_event_based)

stream.parts[0].show()

