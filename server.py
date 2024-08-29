import torch 
import numpy as np
import copy
from sklearn.utils.extmath import randomized_svd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Server():

    def __init__(self,
                 clients,
                 eposide,
                 actor_sparsity,
                 critic_sparsity,
                 update,
                 aggregate) -> None:
        
        self.clients = clients
        self.eposide = eposide
        self.actor_sparisty = actor_sparsity
        self.critic_sparisty = critic_sparsity
        self.update = update
        self.aggregate = aggregate
        
    def fed_model_average(self):
        actors = []
        critics = []
        for client in self.clients:
            actor_model, critic_model = client.get_fed_model_params()
            actors.append(actor_model)
            critics.append(critic_model)
        
        avg_actor_model_params = actors[0].state_dict()
        avg_crtic_model_params = critics[0].state_dict()

        for param_name in avg_actor_model_params:
            for i in range(1, len(actors)):
                avg_actor_model_params[param_name] += actors[i].state_dict()[param_name]
            avg_actor_model_params[param_name] /= len(actors)
        
        for param_name in avg_crtic_model_params:
            for i in range(1, len(critics)):
                avg_crtic_model_params[param_name] += critics[i].state_dict()[param_name]
            avg_crtic_model_params[param_name] /= len(critics)
        
        return avg_actor_model_params, avg_crtic_model_params
    
    def sparse_fed_model_average(self):
        actors_params = []
        critics_params = []
        actors_masks = []
        critics_masks = []
        for client in self.clients:
            actor_model, critic_model = client.get_model_params()
            actors_params.append(actor_model)
            critics_params.append(critic_model)
            actor_masks, critic_masks = client.get_model_masks()
            actors_masks.append(actor_masks)
            critics_masks.append(critic_masks)
        
        actor_count_mask = copy.deepcopy(actors_masks[0])
        critic_count_mask = copy.deepcopy(critics_masks[0])
        avg_actor_model_params = copy.deepcopy(actors_params[0])
        avg_crtic_model_params = copy.deepcopy(critics_params[0])
        
        for k in actor_count_mask.keys():
            actor_count_mask[k] = actor_count_mask[k] - actor_count_mask[k]
            for clnt in range(len(self.clients)):
                actor_count_mask[k] += actors_masks[clnt][k]    
        for k in actor_count_mask.keys():
            actor_count_mask_cpu = actor_count_mask[k].cpu()
            actor_count_mask[k] = np.divide(1, actor_count_mask_cpu, out = np.zeros_like(actor_count_mask_cpu), where = actor_count_mask_cpu != 0)
        for k in avg_actor_model_params.keys():
            avg_actor_model_params[k] = avg_actor_model_params[k] - avg_actor_model_params[k]
            for clnt in range(len(self.clients)):
                avg_actor_model_params[k] += torch.from_numpy(actor_count_mask[k]) * actors_params[clnt][k]
        
        for k in critic_count_mask.keys():
            critic_count_mask[k] = critic_count_mask[k] - critic_count_mask[k]
            for clnt in range(len(self.clients)):
                critic_count_mask[k] += critics_masks[clnt][k]
        for k in critic_count_mask.keys():
            critic_count_mask_cpu = critic_count_mask[k].cpu()
            critic_count_mask[k] = np.divide(1, critic_count_mask_cpu, out = np.zeros_like(critic_count_mask_cpu), where = critic_count_mask_cpu != 0)
        for k in avg_crtic_model_params.keys():
            avg_crtic_model_params[k] = avg_crtic_model_params[k] - avg_crtic_model_params[k]
            for clnt in range(len(self.clients)):
                avg_crtic_model_params[k] += torch.from_numpy(critic_count_mask[k]) * critics_params[clnt][k]
        
        for id, client in enumerate(self.clients):
            local_critic = copy.deepcopy(avg_crtic_model_params)
            for name in critics_masks[id]:
                critic_mask_gpu = critics_masks[id][name].cpu()
                local_critic[name] = local_critic[name] * critic_mask_gpu
            local_actor = copy.deepcopy(avg_actor_model_params)
            for name in actors_masks[id]:
                actor_mask_gpu = actors_masks[id][name].cpu()
                local_actor[name] = local_actor[name] * actor_mask_gpu
            client.set_model_params(local_actor, local_critic)
    
        
    def local_train(self):
        for i in range(self.eposide):
            for id, client in enumerate(self.clients):
                index = i % len(client.train_data)
                reward = client.fed_train(index)
                print(f"home: {id}, eposide: {i}, reward: {reward}")
    
    
    def fed_train(self):
        for eposide in range(self.eposide):        
            for id, client in enumerate(self.clients):
                index = eposide % len(client.train_data)
                reward = client.fed_train(index)
                print(f"home: {id}, eposide: {eposide}, reward: {reward}")    
                        
            if eposide % self.aggregate == 0 and eposide > 0:   
                actor_global_model_params, critic_global_model_params = self.fed_model_average()

                for client in self.clients:
                    client.set_fed_model_params(actor_global_model_params, critic_global_model_params)
            
    
    def sparse_fed_train(self):  
        actor_mask, critic_mask = self.clients[0].initialize(self.actor_sparisty, self.critic_sparisty)
        for client in self.clients:
            client.set_model_masks(actor_mask, critic_mask)
        
        for episode in range(self.eposide):
            if episode % self.aggregate == 0 and episode > 0:
                self.sparse_fed_model_average()
                
            for id, client in enumerate(self.clients):
                index = episode % len(client.train_data)
                reward = client.train(index)
                print(f"home: {id}, eposide: {episode}, reward: {reward}")
                        
            if episode % self.update == 0 and episode > 0:
                for client in self.clients:
                    client.dynamic_update(episode)
        
