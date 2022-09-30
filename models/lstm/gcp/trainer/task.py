import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import random

torch.manual_seed(1)

cuda_availability = torch.cuda.is_available()
if cuda_availability:
  device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
else:
  device = 'cpu'

seq_len = 25
batch_size = 64
lr = 0.001
lstm_layers = 3
lstm_hidden_dim = 512

bucket_path = '/gcs/lstm_maestro/'


def encode_norm(note, transpose):
    return [(note[0]+transpose)/128, note[1]/48]

# Dataset as described in post 3
class DatasetTupelBased(torch.utils.data.Dataset):
    def __init__(self, dir_path, seq_len, device='cpu'):
        #self.data = np.array(np.concatenate([np.load(dir_path+file, allow_pickle=True) for file in os.listdir(dir_path)]))
        self.data = np.load(dir_path, allow_pickle=True)

        self.seq_len = seq_len
        self.device = device

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
        transpose = random.randint(-2, 2)
        x = torch.tensor([encode_norm(note, transpose) for note in part[idx:idx+self.seq_len]], dtype=torch.float32).to(self.device)
        y = torch.tensor([encode_norm(note, transpose) for note in part[idx+1:idx+self.seq_len+1]], dtype=torch.float32).to(self.device)
        return (x, y)

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

def train(dataset, model, num_epochs, loss_fn, opt, batch_size, writer=None, device='cpu', save_dir='data/'):
    if not opt:
        opt = optim.SGD(model.parameters(), lr=0.005)
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    loss_vals = []
    for epoch in range(num_epochs):
        epoch_loss = []
        h_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim).to(device)
        c_t = torch.zeros(model.num_layers, dataset.seq_len, model.hidden_dim).to(device)
        for x, y in dataloader:
            opt.zero_grad()
            y_pred, (h_t, c_t) = model(x, (h_t, c_t))
            loss = loss_fn(y_pred, y)

            if writer:
                writer.add_scalar("Loss", loss, epoch)

            h_t = h_t.detach()
            c_t = c_t.detach()

            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
            print("Epoch loss:", epoch_loss)
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))
        torch.save(model.state_dict(), bucket_path+'models/big_ckp'+str(epoch)+'.pth')
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals)
    plt.title("Loss")
    plt.ticklabel_format(useOffset=False)
    plt.savefig(save_dir+'loss_big.png')
    return sum(loss_vals)

def main():
    dataset = DatasetTupelBased(bucket_path+'test_data/maestro_data.npy', seq_len, device)
    model = LstmModel(2, lstm_hidden_dim, lstm_layers).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    total_loss = train(dataset, model, 15, nn.MSELoss(), opt, batch_size, device=device, save_dir=bucket_path+'img/')
    torch.save(model.state_dict(), bucket_path+'models/big.pth')

if __name__ == '__main__':
    main()
