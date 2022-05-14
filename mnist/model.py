import torch
import torch.nn as nn
import torch.nn.functional as F
class conNet(torch.nn.Module):
  def __init__(self):
    super(conNet, self).__init__()
    self.l1 = nn.Linear(784, 128)
    self.l2 = nn.Linear(128, 64)
    self.l3 = nn.Linear(64, 10)
    nn.init.normal_(self.l1.weight, mean=0, std=1.0)
    nn.init.normal_(self.l2.weight, mean=0, std=1.0)
    nn.init.normal_(self.l3.weight, mean=0, std=1.0)
  def forward(self, x):
    x = self.l1(x)
    x = F.relu(x)
    x = self.l2(x)
    x = F.relu(x)
    x = self.l3(x)
    x = nn.LogSoftmax(dim=1)(x)
    return x
