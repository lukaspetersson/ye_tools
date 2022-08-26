##
import numpy as np
import tqdm
import torch
from torch import nn, optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.manual_seed(1)

SEQ_LEN = 10

##
class Dataset(torch.utils.data.Dataset):
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
    def __init__(self, hidden_dim=20, num_layers=2):
        super(LstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
                input_size=2,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state
##

def train(dataset, model, num_epochs):
    model.train()
    loss_vals = []

    dataloader = DataLoader(dataset, batch_size=1)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = []

        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32)
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
    plt.title("RMSE loss")
    plt.ticklabel_format(useOffset=False)

##
model = LstmModel().to(device=device)
dataset = Dataset("data/data_small.npy", SEQ_LEN)
train(dataset, model, 3)

##
model.eval()
h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32).to(device=device)
c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32).to(device=device)
x = torch.tensor([[[65,2], [65,1], [70,4], [63,2], [68,8]]], dtype=torch.float32).to(device=device)
y_pred, (h_t, c_t) = model(x, (h_t, c_t))
y_pred

##
#TODO
def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words
