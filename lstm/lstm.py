##
import numpy as np
import tqdm
import torch
from torch import nn, optim
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(1)

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
        #TODO: must be float?
        x = torch.tensor(part[idx:idx+self.seq_len], dtype=torch.int8)
        y = torch.tensor(part[idx+1:idx+self.seq_len+1], dtype=torch.int8)
        return (x, y)

##
class LstmModel(nn.Module):
    def __init__(self, hidden_dim=20, num_layers=2):
        super(LstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(2, self.hidden_dim, num_layers=self.num_layers, dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, 2)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state

##
def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=64)
    loss_fn = nn.CrossEntropyLoss()
    #TODO: quasi newton?
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.int8)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.int8)

        for x, y in dataloader:
            opt.zero_grad()
            print(x)
            y_pred, (h_t, c_t) = model(x, (h_t, c_t))
            loss = loss_fn(y_pred, y)

            h_t = h_t.detach()
            c_t = c_t.detach()

            loss.backward()
            opt.step()
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LstmModel().to(device)
dataset = Dataset("data/data_small.npy", SEQ_LEN)
train(dataset, model)
##
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
