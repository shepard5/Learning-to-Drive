import random
import torch
import torch.optim as optim
import torch.nn as nn
from DQN import DQN


class DQN_agent:
    def __init__(self,input_dim,output_dim,learning_rate,gamma=0.99):
            self.device = torch.device("cpu")
            self.q_network = DQN(input_dim,output_dim).to(self.device)
            self.target_network = DQN(input_dim,output_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr = learning_rate)
            self.gamma = gamma
            self.batch_size = 64
            
            ##Training
    def choose_action(self, state, epsilon): 
        if random.uniform(0,1) < epsilon:
            return random.choice(range(self.q_network.fc3.out_features)) #10% probability a random action is selected (exploration)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device) #90% probability max Q action is selected (exploitation)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self,memory):
        if len(memory.memory) <= self.batch_size:
            return
        batch = random.sample(memory.memory, self.batch_size)

        states, actions, rewards, total_rewards, next_states, dones = zip(*batch)
        dones = torch.BoolTensor(dones).to(self.device)
        dones = dones.float()
        states = torch.FloatTensor(states).to(self.device)
        rewards =  torch.FloatTensor(rewards).to(self.device)
        total_rewards = torch.FloatTensor(total_rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions =  torch.LongTensor(actions).to(self.device)

        current_q_values = self.q_network(states).gather(1,actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1.0 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
