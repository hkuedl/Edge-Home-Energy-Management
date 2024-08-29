import torch
from dataset import construct_dataset
from model import Actor, Critic
from agent import P_DQN
from agent_sparse import P_DQN_sparse
from enviroment import home_energy_management
import copy
import math
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

class Client():

    def __init__(
            self,
            data,
            state_dim,
            continous_action_dim,
            discrete_action_dim,
            eposide,
            epsilon) -> None:

        self.data = data
        self.round_max = eposide
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.continous_action_dim = continous_action_dim
        self.discrete_action_dim = discrete_action_dim
        self.train_data, self.test_data = construct_dataset(self.data)

        self.fed_actor_net = Actor(state_dim = self.state_dim, continous_action_dim= self.continous_action_dim).to(device)
        self.fed_critic_net = Critic(state_dim = self.state_dim, continous_action_dim= self.continous_action_dim, discrete_action_dim = self.discrete_action_dim).to(device)
        self.fed_agent = P_DQN(actor_net = self.fed_actor_net, critic_net=self.fed_critic_net, state_dim = self.state_dim, continous_action_dim= self.continous_action_dim, discrete_action_dim = self.discrete_action_dim)
        
        self.actor_net = Actor(state_dim = self.state_dim, continous_action_dim= self.continous_action_dim).to(device)
        self.critic_net = Critic(state_dim = self.state_dim, continous_action_dim= self.continous_action_dim, discrete_action_dim = self.discrete_action_dim).to(device)
        self.agent = P_DQN_sparse(actor_net = self.actor_net, critic_net=self.critic_net, state_dim = self.state_dim, continous_action_dim= self.continous_action_dim, discrete_action_dim = self.discrete_action_dim)

    def initialize(self, actor_sparisty, critic_sparisty):
        actor_params, critic_params = self.agent.get_trainable_params()
        a_sp = self.agent.calculate_sparsities(actor_params, sparse = actor_sparisty)
        c_sp = self.agent.calculate_sparsities(critic_params, sparse = critic_sparisty)
        print(a_sp, c_sp)
        a_mask = self.agent.init_masks(actor_params, a_sp)
        c_mask = self.agent.init_masks(critic_params, c_sp)
        
        return a_mask, c_mask
        
    def train(self, index):
        dataset = self.train_data[index]
        env = home_energy_management(dataset)
        self.state = env.reset()
        epo_reward = 0
        while True:
            a = self.agent.choose_action(self.state)
            s_, _, r, done = env.step(a)
            self.agent.store_transition(self.state, a, r, s_, done)
            self.state = s_
            self.agent.learn()
            epo_reward += r
            if done:
                break
        del env
        return epo_reward 
    
    def test(self):
        dataset = self.test_data
        sum_reward = 0
        sum_reward_elec = 0
        for i in range(len(dataset)):
            env = home_energy_management(dataset[i])
            self.state = env.reset()
            while True:
                a = self.agent.choose_action_test(self.state)
                s_, r_e, r, done = env.step(a)
                self.state = s_
                sum_reward += r
                sum_reward_elec += r_e
                if done:
                    break
            del env
        return sum_reward, sum_reward_elec
    
    def get_model_params(self):
        return self.agent.get_model_params()
    
    def set_model_params(self, actor_params, critic_params):
        self.agent.set_model_params(actor_params, critic_params)
        
    def get_model_masks(self):
        return self.agent.get_model_masks()
    
    def set_model_masks(self, actor_masks, critic_masks):
        self.agent.set_model_masks(actor_masks, critic_masks)
        
    def fire_mask(self, weights, masks, round):
        drop_ratio = self.epsilon / 2 * (1 + np.cos((round * np.pi) / self.round_max))
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            weights_gpu = weights[name].to(device)
            temp_weights = torch.where(masks[name] > 0, torch.abs(weights_gpu), 100000 * torch.ones_like(weights_gpu))
            _, idx = torch.sort(temp_weights.view(-1).to(device))
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return new_masks, num_remove

    def regrow_mask(self, masks, num_remove, gradient):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            gradient_gpu = gradient[name].to(device)
            temp = torch.where(masks[name] == 0, torch.abs(gradient_gpu), -100000 * torch.ones_like(gradient_gpu))
            _, idx = torch.sort(temp.view(-1).to(device), descending=True)
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return new_masks
    
    def dynamic_update(self, round):
        actor_weights, critic_weights = self.agent.get_model_params()
        actor_masks, critic_masks = self.agent.get_model_masks()
        actor_gradient, critic_gradient = self.agent.screen_gradients()
        actor_new_masks, actor_num_remove = self.fire_mask(actor_weights, actor_masks, round)
        actor_new_masks = self.regrow_mask(actor_new_masks, actor_num_remove, actor_gradient)
        critic_new_masks, critic_num_remove = self.fire_mask(critic_weights, critic_masks, round)
        critic_new_masks = self.regrow_mask(critic_new_masks, critic_num_remove, critic_gradient)
        self.agent.set_model_masks(actor_new_masks, critic_new_masks)
        
    def fed_train(self, index):
        dataset = self.train_data[index]
        env = home_energy_management(dataset)
        self.state = env.reset()
        epo_reward = 0
        while True:
            a = self.fed_agent.choose_action(self.state)
            s_, _, r, done = env.step(a)
            self.fed_agent.store_transition(self.state, a, r, s_, done)
            self.state = s_
            self.fed_agent.learn()
            epo_reward += r
            if done:
                break
        del env
        return epo_reward 
    
    def fed_test(self):
        dataset = self.test_data
        sum_reward = 0
        sum_reward_elec = 0
        for i in range(len(dataset)):
            env = home_energy_management(dataset[i])
            self.state = env.reset()
            while True:
                a = self.fed_agent.choose_action_test(self.state)
                s_, r_e, r, done = env.step(a)
                self.state = s_
                sum_reward += r
                sum_reward_elec += r_e
                if done:
                    break
            del env
        return sum_reward, sum_reward_elec
    
    def get_fed_model_params(self):
        return self.fed_agent.get_model_params()
    
    def set_fed_model_params(self, actor_params, critic_params):
        self.fed_agent.set_model_params(actor_params, critic_params)
    
