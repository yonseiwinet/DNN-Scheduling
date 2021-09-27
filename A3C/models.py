import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        value = F.relu(self.fc3(value))
        value = self.fc4(value)
        return value


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, output_dim)
    
    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = F.relu(self.fc2(logits))
        logits = F.relu(self.fc3(logits))
        logits = self.fc4(logits)
        return logits