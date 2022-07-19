##
import torch
from torch import nn, optim
from torch.nn.functional import relu
import matplotlib.pyplot as plt

##
def complex_relu(z):
    return relu(z.real) + 1.j * relu(z.imag)

class CvnnTest(nn.Module):
    def __init__(self):
        super(CvnnTest, self).__init__()
        self.l1 = nn.Linear(1, 5).to(torch.cfloat)
        self.l2 = nn.Linear(5, 5).to(torch.cfloat)
        self.l3 = nn.Linear(5, 1).to(torch.cfloat)
        self.relu = complex_relu

    def forward(self, inputs):
        x = self.l1(inputs)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

##
def func(z):
    return (2*z.real+5) + 1.j*(3*z.imag-3) 

x = torch.randn((1000,1), dtype=torch.cfloat)
y = x.clone().detach()
y.apply_(func) 
plt.plot([z.real for z in x], [z.real for z in y], 'ro')
plt.plot([z.imag for z in x], [z.imag for z in y], 'bo')

##
model = CvnnTest()
model.train()
opt = optim.Adam(model.parameters())
loss_fn = nn.L1Loss()

loss_vals = []
for _ in range(1000):
    opt.zero_grad()
    x = torch.randn((1000,1), dtype=torch.cfloat)
    y = x.clone().detach()
    y.apply_(func) 
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    opt.step()
    loss_vals.append(loss.item())
plt.plot(loss_vals)

##
model.eval()
x = torch.randn((1000,1), dtype=torch.cfloat)
y = model(x)
plt.plot([z.real for z in x], [z.real for z in y], 'ro')
plt.plot([z.imag for z in x], [z.imag for z in y], 'bo')

