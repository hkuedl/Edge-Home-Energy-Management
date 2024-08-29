import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from model import Actor, Critic
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)
        
        
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = np.stack(state_batch)
        next_state_batch = np.stack(next_state_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save(self, filename):
        with open(filename, 'w') as f:
            for item in self.buffer:
                f.write(str(item) + '\n')
                    
    def __len__(self):
        return len(self.buffer)

class P_DQN(object):
    def __init__(self,
                 actor_net: nn.Module,
                 critic_net: nn.Module,
                 discrete_action_dim,
                 continous_action_dim,
                 state_dim):
        
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.state_dim = state_dim
        self.continous_action_dim = continous_action_dim
        self.discrete_action_dim = discrete_action_dim
        self.actor_target_net = Actor(self.state_dim, self.continous_action_dim).to(device)
        self.critic_target_net = Critic(self.state_dim, self.continous_action_dim, self.discrete_action_dim).to(device)

        hard_update_target_network(self.actor_net, self.actor_target_net)
        hard_update_target_network(self.critic_net, self.critic_target_net)

        self.memory_capacity = 1000
        self.continous_action_min = [-2.5, 0, -2.4]
        self.continous_action_max = [2.5, 6, 2.4]
        self.gamma = 0.99
        self.batch_size = 32
        self.lr_actor = 0.00001
        self.lr_critic = 0.0001
        self.epsilon_start = 1
        self.epsilon_end = 0.005
        self.epsilon_decay = 10000
        self.learn_step_counter = 0
        self.critic_tau = 0.01
        self.actor_tau = 0.001
        self.memory = ReplayBuffer(self.memory_capacity)

        self.frame_idx = 0 
        self.epsilon = lambda frame_idx: self.epsilon_end + \
                                         (self.epsilon_start - self.epsilon_end) * \
                                         math.exp(-1. * frame_idx / self.epsilon_decay)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor) 
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic) 

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                continous_action = self.actor_net(state)
                q_values = self.critic_net(state, continous_action)
                q_values = q_values.detach().cpu().data.numpy()
                discrete_action = q_values.argmax().item()
            continous_action = continous_action.squeeze(0)
        else:
            discrete_action = random.randrange(self.discrete_action_dim)
            continous_action = torch.tensor(np.random.uniform(self.continous_action_min,
                                                              self.continous_action_max,
                                                              size = self.continous_action_dim)).to(device)
        continous_action = continous_action.cpu().numpy()
        # print(f"discrete_action: {discrete_action}, continous_action: {continous_action}")
        action = (discrete_action, continous_action)
        return action

    def choose_action_test(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            continous_action = self.actor_net(state)
            q_values = self.critic_net(state, continous_action)
            q_values = q_values.detach().cpu().data.numpy()
            discrete_action = q_values.argmax().item()
        continous_action = continous_action.squeeze(0)
        continous_action = continous_action.cpu().numpy()
        action = (discrete_action, continous_action)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.from_numpy(state_batch).float().to(device)
        discrete_action_batch = [a[0] for a in action_batch]
        continous_action_batch = [a[1] for a in action_batch]
        discrete_action_batch = torch.tensor(discrete_action_batch).unsqueeze(1).to(device)
        continous_action_batch = np.array(continous_action_batch)
        continous_action_batch = torch.from_numpy(continous_action_batch).float().to(device)
        reward_batch = torch.from_numpy(reward_batch).float().to(device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(device)
        done_batch = torch.from_numpy(done_batch).float().to(device)
        
        # Update critic netowork
        with torch.no_grad():
            next_continous_action_batch = self.actor_target_net(next_state_batch)
            next_q_values = self.critic_target_net(next_state_batch, next_continous_action_batch)
            next_q_values_max = next_q_values.max(1)[0].detach()
            target = reward_batch + self.gamma * next_q_values_max * (1 - done_batch)
        
        q_values = self.critic_net(state_batch, continous_action_batch) 
        q_values = q_values.gather(1, index=discrete_action_batch)
        loss_critic = nn.MSELoss()(q_values, target.unsqueeze(1))
        self.critic_net.train()
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        # Update actor netowork
        update_continous_action_batch = self.actor_net(state_batch)
        update_q_values = self.critic_net(state_batch, update_continous_action_batch)
        loss_critic = -torch.mean(update_q_values)
        
        # Calculate the gradient of the critic network with respect to the continous action
        self.actor_net.train()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss_critic.backward(retain_graph=True)
        self.actor_optimizer.step() 
        
        soft_update_target_network(self.actor_net, self.actor_target_net, self.actor_tau)
        soft_update_target_network(self.critic_net, self.critic_target_net, self.critic_tau)
    
    def set_model_params(self, actor_model, critic_model):
        self.actor_net.load_state_dict(actor_model)
        self.critic_net.load_state_dict(critic_model)
    
    def get_model_params(self):
        return self.actor_net, self.critic_net
 