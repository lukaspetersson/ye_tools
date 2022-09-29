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
import music21 as m21
from torch.utils.tensorboard import SummaryWriter
from utils import * 

torch.manual_seed(1)
device = "cpu"
SEQ_LEN = 25
writer = SummaryWriter()


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
model_tup = LstmModel(input_dim=2).to(device)
model_tup.load_state_dict(torch.load('models/gcp/test.pth'))

##
start = torch.tensor(np.load('data/twinkle_note_based.npy', allow_pickle=True)[0][:SEQ_LEN], dtype=torch.float)
seq = generate(model_tup, start, len(start)*2, SEQ_LEN)
stream = to_stream_tup_based(seq)
stream.write('midi', fp='test.midi')

##
start = [375, 60, 371, 52, 372, 48, 315, 188, 258, 376, 60, 312, 180, 176, 258, 188, 258, 378, 67, 375, 60, 373, 48, 315, 195, 258, 377, 67, 312, 188, 176, 258, 195, 258, 377, 69, 374, 60, 372, 41, 315, 197, 258, 376, 69, 312, 188, 169, 258, 197, 258, 376, 67, 374, 60, 373, 48, 355, 275, 195, 188, 176, 261, 376, 65, 373, 57, 373, 41, 315, 193, 258, 377, 65, 312, 185, 169, 258, 193, 258, 376, 64, 373, 55, 375, 48, 315, 192, 258, 377, 64, 312, 183, 176, 258, 192, 258, 376, 62, 373, 55, 373, 43, 315, 190, 258, 376, 62, 312, 183, 171, 258, 190, 258, 376, 60, 373, 52, 374, 48, 355, 275, 188, 180, 176, 261, 379, 67, 376, 60, 374, 48, 315, 195, 258, 376, 67, 312, 188, 176, 258, 195, 258, 376, 65, 373, 57, 373, 41, 315, 193, 258, 377, 65, 312, 185, 169, 258, 193, 258, 376]
start = list(filter(lambda x: x<356, start))
start = F.one_hot(torch.tensor(start), num_classes=356).float()
seq_event_based = generate(model_event, start, 2*len(start), SEQ_LEN)
stream = to_stream_event_based(seq_event_based)

stream.parts[0].show()

