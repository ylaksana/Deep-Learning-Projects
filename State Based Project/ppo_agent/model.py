import torch
import torch.nn.functional as F
import torch.nn as nn


class FeedForwardNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output

def save_model(model, id):
    model.eval()
    model = model.to('cpu')
    scripted_model = torch.jit.script(model)
    return scripted_model.save(f'state_agent/{id}.jit')

def load_model(id):
    from torch import load
    from os import path
    device = torch.device('cpu')
    model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'{id}.jit'), map_location=device)
    return model
