##
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(1)

##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len):
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
        x = torch.tensor(part[i:i+self.seq_len], dtype=torch.int8).float()
        y = torch.tensor(part[i+1:i+self.seq_len+1], dtype=torch.int8).float()
        return (x, y)

##
dataset = Dataset("data/data.npy", 10)
dataset[100][1].size()

##
class LstmModel(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2):
        super(LstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(128, self.hidden_dim, num_layers=self.num_layers, dropout=0.1)
        self.linear = nn.Linear(self.hidden_dim, 128)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        pred = self.linear(output)
        return pred, state
model = Model()

##
def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=64)
    loss_fn = nn.CrossEntropyLoss()
    #TODO: quasi newton?
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim, dtype=torch.float32)

        for x, y in tqdm(dataloader):
            optimizer.zero_grad()
            print(x.dtype, y.dtype)

            y_pred, (h_t, c_t) = model(x, (h_t, c_t))
            #TODO: dimensions
            loss = loss_fn(y_pred, y)

            h_t = h_t.detach()
            c_t = c_t.detach()

            loss.backward()
            optimizer.step()
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
