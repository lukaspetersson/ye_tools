##
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(1)

##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, seq_len):
        self.data = np.load(data_dir, allow_pickle=True)
        self.seq_len = seq_len

    def __len__(self):
        l = 0
        for song in self.data:
            l += (len(song) - self.seq_len)
        return l

    def get_song(self, idx):
        i = idx
        for song in self.data:
            i -= (len(song) - self.seq_len)
            if i < 0:
                break
        return (song, i)

    def __getitem__(self, idx):
        song, i = get_song(idx)
        x = torch.tensor(song[i-self.seq_len:i])
        y = torch.tensor(song[i-self.seq_len+1:i+1])
        return (x, y)
