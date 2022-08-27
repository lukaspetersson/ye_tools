##
import numpy as np
import glob
import tqdm
import torch
from torch import nn, optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from encode_midi import *

torch.manual_seed(1)

SEQ_LEN = 10

##

class DatasetEventBased(torch.utils.data.Dataset):
    def __init__(self, file_paths, seq_len):
        self.data = [encode_midi(file_path) for file_path in file_paths] 
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
        x = F.one_hot(torch.tensor(part[idx:idx+self.seq_len]), num_classes=388).float()
        y = F.one_hot(torch.tensor(part[idx+1:idx+self.seq_len+1]), num_classes=388).float()
        return (x, y)

##
class DatasetNoteBased(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len=SEQ_LEN):
        self.data = np.load(data_dir, allow_pickle=True)
        #self.data = np.array(np.concatenate([np.load('data/data_big_'+str(i*100)+'.npy', allow_pickle=True) for i in range(1,data_size)]))

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
        x = torch.tensor(part[idx:idx+self.seq_len], dtype=torch.float32)
        y = torch.tensor(part[idx+1:idx+self.seq_len+1], dtype=torch.float32)
        return (x, y)
##
device = "cuda"

##
class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, num_layers=2):
        super(LstmModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state
##
def train(dataset, model, num_epochs, loss_fn):
    model.train()
    loss_vals = []

    dataloader = DataLoader(dataset, batch_size=64)
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = []

        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim)
        for x, y in dataloader:
            opt.zero_grad()
            y_pred, (h_t, c_t) = model(x.to(device=device), (h_t.to(device=device), c_t.to(device=device)))
            loss = loss_fn(y_pred.to(device=device), y.to(device=device))

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
model_event = LstmModel(input_dim=388).to(device=device)
dataset_event = DatasetEventBased(file_paths=list(glob.glob('data/lmd_full/0/*.mid', recursive=True))[:1], seq_len=SEQ_LEN*10)
train(dataset_event, model_event, 3, nn.CrossEntropyLoss())

##
model_note = LstmModel(input_dim=2).to(device=device)
dataset_note = DatasetNoteBased('data/data_big_100.npy', SEQ_LEN)
train(dataset_note, model_note, 3, nn.MSELoss())

##
model_event.eval()
h_t = torch.zeros(model_event.num_layers, dataset_event.seq_len, model_event.hidden_dim).to(device=device)
c_t = torch.zeros(model_event.num_layers, dataset_event.seq_len, model_event.hidden_dim).to(device=device)
x = torch.unsqueeze(dataset_event[1500][0].to(device=device), dim=0)

y_pred, state = model_event(x, (h_t, c_t))
y_pred[0][-1]

##
model_note.eval()
h_t = torch.zeros(model_note.num_layers, dataset_note.seq_len, model_note.hidden_dim).to(device=device)
c_t = torch.zeros(model_note.num_layers, dataset_note.seq_len, model_note.hidden_dim).to(device=device)
x = torch.unsqueeze(dataset_note[1500][0].to(device=device), dim=0)

y_pred, state = model_note(x, (h_t, c_t))
y_pred[0][-1]

##
def generate(model, start, song_len, seq_len):
    model.eval()
    start_len = len(start)
    seq = torch.cat((start, torch.zeros(song_len-start_len, model.input_dim, dtype=torch.float))).to(device=device)
    h_t = torch.zeros(model.num_layers, seq_len, model.hidden_dim).to(device=device)
    c_t = torch.zeros(model.num_layers, seq_len, model.hidden_dim).to(device=device)
    for i in range(start_len, song_len):
        x = torch.unsqueeze(seq[i-seq_len:i], dim=0)
        y, (h_t, c_t) = model(x, (h_t, c_t))
        seq[i] = y[0][-1]
    return seq

##

def to_m21_note_based(note):
    return m21.note.Note(pitch=round(note[0].item()), duration=m21.duration.Duration(round(note[1].item()/4)))

def to_m21_event_based(note):
    return None

def to_stream_note_based(seq, to_m21_func):
    stream = m21.stream.Stream()
    for note in seq:
        stream.append(to_m21_func(note))
    return stream

##
start = torch.tensor(np.load('data/twinkle_note_based.npy', allow_pickle=True)[0][:16], dtype=torch.float)
seq_note_based = generate(model_note, start, 30, SEQ_LEN)
stream = to_stream_note_based(seq_note_based, to_m21_note_based)
stream.show()

##
start = F.one_hot(torch.tensor([375, 60, 371, 52, 372, 48, 315, 188, 258, 376, 60, 312, 180, 176, 258, 188, 258, 378, 67, 375, 60, 373, 48, 315, 195, 258, 377, 67, 312, 188, 176, 258, 195, 258, 377, 69, 374, 60, 372, 41, 315, 197, 258, 376, 69, 312, 188, 169, 258, 197, 258, 376, 67, 374, 60, 373, 48, 355, 275, 195, 188, 176, 261, 376, 65, 373, 57, 373, 41, 315, 193, 258, 377, 65, 312, 185, 169, 258, 193, 258, 376, 64, 373, 55, 375, 48, 315, 192, 258, 377, 64, 312, 183, 176, 258, 192, 258, 376, 62, 373, 55, 373, 43, 315, 190, 258, 376, 62, 312, 183, 171, 258, 190, 258, 376, 60, 373, 52, 374, 48, 355, 275, 188, 180, 176, 261, 379, 67, 376, 60, 374, 48, 315, 195, 258, 376, 67, 312, 188, 176, 258, 195, 258, 376, 65, 373, 57, 373, 41, 315, 193, 258, 377, 65, 312, 185, 169, 258, 193, 258, 376]), num_classes=388).float()
#seq_event_based = generate(model_event, start, 30*10, SEQ_LEN*10)
#stream = to_stream_note_based(seq_note_based, to_m21_event_based)
#stream.show()

def decode_midi(idx_array, file_path=None):
    event_sequence = [Event.from_int(idx) for idx in idx_array]
    # print(event_sequence)
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)
    print(note_seq)
##
decode_midi(start)

