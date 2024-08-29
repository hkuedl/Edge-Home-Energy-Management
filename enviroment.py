import numpy as np

class home_energy_management(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.max_step = len(self.dataset)
        self.current_step = 0
        # factor of thermal dynamic
        self.epsilon = 0.7
        self.lamda = 10
        # max_duration of washing machine
        self.max_duration = 2 
        self.identify = 0
        self.rated_power_WM = 0.7
        # max power of ac, ev and ess
        self.max_power_AC = 2.5
        self.max_power_EV = 6
        self.max_power_ESS = 2.4
        # max capacity of EV and ESS
        self.max_capacity_EV = 24
        self.max_capacity_ESS = 6.4
        # charging and discharging efficiency of EV and ESS
        self.efficiency_EV = 0.95
        self.efficiency_ESS = 0.95
        # initial and end time of EV
        self.time_ini_EV = 0
        self.time_end_EV = 8
        # penality factor in reward
        self.alpha = 0.005
        self.beta = 0.008
        self.mu = 0.05
        self.sigma = 0.01
        # prefer time in reward
        self.time_ini_WM = 10
        self.time_end_WM = 20
        # prefer temperature in reward
        self.temperature_min = 20
        self.temperature_max = 26
        # discrete action space of washing machine: 0: no action, 1: start
        self.discrete_action_space = 2
        # continuous action space of ac, ev and ess
        self.continuous_action_space = 3
        self.state_space = 9
        # state = [hour, price, PV, fixed_load, c_WM, T_in, T_out, SoE_EV, SoE_ESS]
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.identify = 0
        self.state = np.zeros(self.state_space)
        self.state[0] = 0
        self.state[1] = self.dataset[self.current_step][3]
        self.state[2] = self.dataset[self.current_step][1]
        self.state[3] = self.dataset[self.current_step][0]
        self.state[4] = 0
        self.state[5] = 23 / 20
        self.state[6] = self.dataset[self.current_step][4]
        self.state[7] = 0.5
        self.state[8] = 0.2
        return self.state

    def step(self, action):
        discrete_action = action[0]
        continuous_action = action[1]
        self.state = self.state_transition(discrete_action, continuous_action)
        reward, reward_elec = self.calculate_reward()
        done = self.check_termination()
        return self.state, reward_elec, reward, done 

    def state_transition(self, discrete_action, continuous_action):
        self.current_step += 1
        self.price = self.state[1] / 20
        self.pv_generation = self.state[2]
        self.fixed_load = self.state[3]
        self.identify = 1 if (self.state[4] == 0 and discrete_action == 1) or (self.identify == 1 and self.state[4] < self.max_duration) else 0
        self.power_WM = self.identify * self.rated_power_WM
        self.power_AC = np.clip(continuous_action[0], -self.max_power_AC, self.max_power_AC)
        self.indoor_temp = self.state[5] * 20
        self.outdoor_temp = self.state[6] * 20
        self.expected_temp = self.epsilon * self.indoor_temp + (1-self.epsilon) * (self.outdoor_temp + self.lamda * self.power_AC)
        
        if self.current_step - 1 < self.time_end_EV:
            self.power_EV = np.clip(continuous_action[1], 0, self.max_power_EV)
            self.SoE_EV = self.state[7] + self.efficiency_EV * self.power_EV / self.max_capacity_EV
            if self.SoE_EV > 1:
                self.power_EV = (1 - self.state[7]) * self.max_capacity_EV / self.efficiency_EV
        else:
            self.power_EV = 0
        
        self.power_ESS = np.clip(continuous_action[2], -self.max_power_ESS, self.max_power_ESS)
        self.SoE_ESS = self.state[8] + self.efficiency_ESS * self.power_ESS / self.max_capacity_ESS
        if self.SoE_ESS > 1:
            self.power_ESS = (1 - self.state[8]) * self.max_capacity_ESS / self.efficiency_ESS
        elif self.SoE_ESS < 0:
            self.power_ESS = -self.state[8] * self.max_capacity_ESS / self.efficiency_ESS
            
        new_state = np.zeros(self.state_space)
        if self.current_step >= self.max_step:
            new_state[0] = 24 / 20
            new_state[1] = 0
            new_state[2] = 0
            new_state[3] = 0
            new_state[6] = 23 / 20
        else:
            new_state[0] = self.dataset[self.current_step][2]
            new_state[1] = self.dataset[self.current_step][3]
            new_state[2] = self.dataset[self.current_step][1]
            new_state[3] = self.dataset[self.current_step][0]
            new_state[6] = self.dataset[self.current_step][4]
        new_state[4] = self.state[4] + self.identify
        new_state[5] = self.expected_temp / 20
        new_state[7] = self.state[7] + self.efficiency_EV * self.power_EV / self.max_capacity_EV
        new_state[8] = self.state[8] + self.efficiency_ESS * self.power_ESS / self.max_capacity_ESS
        return new_state

    def calculate_reward(self):
        reward = 0
        elec = self.fixed_load + self.power_WM + abs(self.power_AC) + self.power_EV + self.power_ESS - self.pv_generation
        if elec > 0:
            reward_elec = -elec * self.price
        else:
            reward_elec = 0
        
        # reward for use of washing machine
        if self.identify == 1:
            if self.current_step - 1 < self.time_ini_WM:
                reward_pre = -self.alpha * (self.time_ini_WM - (self.current_step - 1)) **2
            elif self.current_step - 1 > self.time_end_WM:
                reward_pre = -self.alpha * ((self.current_step - 1) - self.time_end_WM) **2
            else:
                reward_pre = 0
        else:
            reward_pre = 0
            
        # reward for temperature of indoor
        if self.expected_temp < self.temperature_min:
            reward_temp = -self.beta * (self.temperature_min - self.expected_temp) **2   
        elif self.expected_temp > self.temperature_max:
            reward_temp = -self.beta * (self.expected_temp - self.temperature_max) **2
        else: 
            reward_temp = 0
        
        # reward for travel of EV
        if self.current_step == self.time_end_EV:
            reward_travel = -self.mu * (1 - self.state[7])
        else:
            reward_travel = 0
            
        # reward for times of using washing machine
        if self.current_step == self.max_step:
            reward_use = -self.sigma * (self.max_duration - self.state[4]) **2
        else:
            reward_use = 0
        
        reward = reward_elec + reward_pre + reward_temp + reward_travel + reward_use
        return reward, reward_elec

    def check_termination(self):
        if self.current_step >= self.max_step:
            done = True
        else:
            done = False
        
        return done