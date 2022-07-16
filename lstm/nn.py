##
import torch
from torch import nn, optim
from torch.nn.functional import relu

##
def complex_relu(z):
    return relu(z.real) + 1.j * relu(z.imag)

class ComplexTest(nn.Module):
    def __init__(self):
        super(ComplexTest, self).__init__()
        self.l1 = nn.Linear(5, 10).to(torch.cfloat)
        self.l2 = nn.Linear(10, 5).to(torch.cfloat)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexTest().to(device)
model.train()
opt = optim.Adam(model.parameters())
loss_fn = nn.L1Loss()

##
y = torch.view_as_complex(torch.tensor([0.5, 0.1], dtype=torch.float))
for _ in range(500):
    inputs = torch.randn((1000, 5), dtype=torch.cfloat)
    y_pred = model(inputs)
    loss = loss_fn(y_pred, y)
    loss.backward()
    opt.step()

##
x = torch.randn((5,), dtype=torch.cfloat)
model(x)
