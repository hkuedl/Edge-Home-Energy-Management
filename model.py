import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_size = 256
# import gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, continous_action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, continous_action_dim) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        a = self.fc4(x)
        return a
    
class Critic(nn.Module):
    def __init__(self, state_dim, continous_action_dim, discrete_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + continous_action_dim, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, discrete_action_dim) 

    def forward(self, state, continous_action):
        x = torch.cat((state, continous_action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        return q


    
