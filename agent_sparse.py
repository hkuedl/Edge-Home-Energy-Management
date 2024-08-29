import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import copy
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

class P_DQN_sparse(object):
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
    
    
    def init_masks(self, params, sparsities):
        masks ={}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] =1
        return masks

    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse = 0.5):
        spasities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - sparse
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            total_params = 0
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            dense_layers = set()

            
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * sparse
                    n_ones = n_param * (1 - sparse)

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(params[name].shape) / np.prod(params[name].shape)
                                                  )
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            for name in params:
                if name in dense_layers:
                    spasities[name] = 0
                else:
                    spasities[name] = (1 - epsilon * raw_probabilities[name])
        return spasities
    
    def choose_action(self, state):
        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)
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
        action = (discrete_action, continous_action)
        return action

    def choose_action_test(self, state):
        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)
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

        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)
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
        
        for name, param in self.critic_net.named_parameters():
            if name in self.critic_masks:
                param.data *= self.critic_masks[name].to(device)
        
        # Update actor netowork
        update_continous_action_batch = self.actor_net(state_batch)
        update_q_values = self.critic_net(state_batch, update_continous_action_batch)
        loss_critic = -torch.mean(update_q_values)
        
        self.actor_net.train()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        loss_critic.backward(retain_graph=True)
        self.actor_optimizer.step() 
        for name, param in self.actor_net.named_parameters():
            if name in self.actor_masks:
                param.data *= self.actor_masks[name].to(device)
        
        soft_update_target_network(self.actor_net, self.actor_target_net, self.actor_tau)
        soft_update_target_network(self.critic_net, self.critic_target_net, self.critic_tau)

        for name, param in self.critic_target_net.named_parameters():
            if name in self.critic_masks:
                param.data *= self.critic_masks[name].to(device)
                
        for name, param in self.actor_target_net.named_parameters():
            if name in self.actor_masks:
                param.data *= self.actor_masks[name].to(device)
    
    def get_model_params(self):
        return copy.deepcopy(self.actor_net.cpu().state_dict()), copy.deepcopy(self.critic_net.cpu().state_dict())

    def get_trainable_params(self):
        actor_dict= {}
        critic_dict = {}
        for name, param in self.actor_net.named_parameters():
            actor_dict[name] = param
        for name, param in self.critic_net.named_parameters():
            critic_dict[name] = param
        return actor_dict, critic_dict
    
    def set_model_params(self, actor_parameters, critic_parameters):
        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)
        self.actor_net.load_state_dict(actor_parameters)
        self.critic_net.load_state_dict(critic_parameters)
    
    def get_model_masks(self):
        return copy.deepcopy(self.actor_masks), copy.deepcopy(self.critic_masks)
    
    def set_model_masks(self, actor_masks, critic_masks):
        self.actor_masks = actor_masks
        self.critic_masks = critic_masks
  	
    def screen_gradients(self):
        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)
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
        
        self.actor_net.eval()
        self.critic_net.eval()

        next_continous_action_batch = self.actor_target_net(next_state_batch)
        next_q_values = self.critic_target_net(next_state_batch, next_continous_action_batch)
        next_q_values_max = next_q_values.max(1)[0].detach()
        target = reward_batch + self.gamma * next_q_values_max * (1 - done_batch)
        
        q_values = self.critic_net(state_batch, continous_action_batch) 
        q_values = q_values.gather(1, index=discrete_action_batch)
        loss_critic = nn.MSELoss()(q_values, target.unsqueeze(1))

        critic_gradient={}
        self.critic_net.zero_grad()
        loss_critic.backward()
        for name, param in self.critic_net.named_parameters():
            critic_gradient[name] = param.grad.cpu()
        
        # Update actor netowork
        update_continous_action_batch = self.actor_net(state_batch)
        update_q_values = self.critic_net(state_batch, update_continous_action_batch)
        loss_critic = -torch.mean(update_q_values)
        
        actor_gradient={}
        self.critic_net.zero_grad()
        self.actor_net.zero_grad()
        loss_critic.backward(retain_graph=True)
        for name, param in self.actor_net.named_parameters():
            actor_gradient[name] = param.grad.cpu()
            
        return actor_gradient, critic_gradient
        