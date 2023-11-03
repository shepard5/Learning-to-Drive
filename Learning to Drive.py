import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

#rows, cols = 300, 300
#track_array = np.zeros((rows,cols))

class Track :
    def __init__(self, rows = 100, cols = 100):
         self.rows = rows
         self.cols = cols
         self.track = np.zeros((self.rows, self.cols))
         self.setup_track()
         
    def setup_track(self):
        x_points = self.cols
        x = np.linspace(0,2*np.pi, x_points)
    
        #y_upper = 0.375*self.rows*np.sin(1.5*x) + 750
        y = int(self.rows*.8)*np.sin(.5*x) 


        grid = np.zeros((self.rows,self.cols))
            
        for i in range(x_points):
            grid[int(y[i]):int(y[i])+10,i] = 1
            
            #lower_bound = int((y_lower[i] - min(y_lower)) )  # convert to grid index
            #upper_bound = int((y_upper[i] - min(y_lower)) )  # convert to grid index
            #grid[lower_bound:upper_bound, i] = 1  # mark as track
            #grid[lower_bound+50:upper_bound-50, i] = 0
            if  95<i<100:
                grid[int(y[i]):int(y[i])+10,i] = 2
            
        self.track = grid
    
    def display(self):
        plt.imshow(self.track, origin="lower", cmap="gray_r")
        plt.title("Track Visualization")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        
    def __getitem__(self, index):
        # Implement the __getitem__ method to allow subscripting
        x, y = index
        return self.track[x][y]
        
class Car: 
    def __init__(self, angle = 0):
        self.velocity = 0
        
        self.x = 0
        self.y = 0
        self.x_grid = round(self.x)
        self.y_grid = round(self.y)
        
        self.angle = angle
        
        self.max_speed = 10
        self.acceleration = 3
        self.turn_rate = 5.0
        
        self.has_finished = False
        self.has_crashed = False
        self.last_position = None
        self.distance_traveled = np.sqrt(self.x*self.x+self.y*self.y)
    def drive(self):
        self.velocity = self.velocity + self.acceleration
    
    def brake(self):
        self.velocity = self.velocity - self.acceleration
    
    def turn_left(self):
        self.angle += self.turn_rate
        if self.angle >= 360:
            self.angle += 360
    
    def turn_right(self):
        self.angle -= self.turn_rate
        if self.angle < 0:
            self.angle += 360
    
    def move(self, track):
        rad_angle = np.radians(self.angle)
        self.x += self.velocity * np.cos(rad_angle)
        self.y += self.velocity * np.sin(rad_angle)
        
        if self.x <= 0:
            car.has_crashed == True
        if track[self.x_grid,self.y_grid] == 0:
            car.has_crashed = True
        
        elif track[self.x_grid,self.y_grid] == 2:
            car.has_finished = True
    
    def step(self,action,track):
        if action == 'ACCELERATE' or action == 0:
            self.drive()
        elif action == 'BRAKE' or action == 1:
            self.brake()
        elif action == 'TURN_LEFT' or action == 2:
            self.turn_left()
        elif action == 'TURN_RIGHT' or action == 3:
            self.turn_right()
            
        self.move(track)
        
    def get_state(self):
        return self.x_grid, self.y_grid, self.angle, self.velocity
    
    def choose_action(state, model, epsilon):
    # If a randomly chosen value is less than epsilon, choose a random action (Exploration)
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(model.fc3.out_features))
        else:
            # Convert the state to a tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert state to tensor and add batch dimension
            
            # Get predicted Q-values from the model for the current state
            with torch.no_grad():
                q_values = model(state_tensor)
            
            # Choose the action with maximum Q-value (Exploitation)
            return torch.argmax(q_values).item()
        
    def get_reward(self):
        reward = 0
        if self.has_crashed:
            return -100
        elif self.has_finished:
            return 5000
        if self.velocity < 5:
            reward -= 5
        elif self.velocity > 10:
            reward += 10
#        reward += self.distance_traveled/300
        reward -= 2
        return reward

class ReplayBuffer:
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Overwrite old data when capacity is reached

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
ACTIONS = ['ACCELERATE','BRAKE','TURN_LEFT','TURN_RIGHT']
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, len(ACTIONS))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def choose_action(state, model, epsilon):
    # If a randomly chosen value is less than epsilon, choose a random action (Exploration)
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(model.fc3.out_features))
    else:
        # Convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert state to tensor and add batch dimension
        
        # Get predicted Q-values from the model for the current state
        with torch.no_grad():
            q_values = model(state_tensor)
        
        # Choose the action with maximum Q-value (Exploitation)
        return torch.argmax(q_values).item()


track = Track()

input_dim = 4
num_actions = 4
learning_rate = 0.001
gamma = 0.99

model = DQN(input_dim, num_actions)

batch_size = 50
memory = ReplayBuffer(capacity = 10000)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.995



for episode in range(500):
    car = Car()
    state = car.get_state()
    done = False
    reward_total = 0
    ep_it = 0
    while not done and ep_it<20:
        action = choose_action(state, model, epsilon)
        
        # Execute action using Car class, get next state, reward, etc. from Track class
        #next_state, reward, done, _ = ...
        print(action)
        car.step(action,track)
        next_state = car.get_state()
        print(next_state)
        reward_total += car.get_reward()    
        if car.has_crashed or car.has_finished:
            done = True
        
        memory.push(state, action, reward_total, next_state, done)
        
        if len(memory) > batch_size:
            experiences = memory.sample(batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*experiences)
            batch_states = torch.tensor(batch_states, dtype=torch.float32)
            batch_actions = torch.tensor(batch_actions, dtype=torch.int64)
        
        
        # Get current Q-values
        q_values = model(torch.tensor(state, dtype=torch.float32))
        
        # Compute the target Q-value
        with torch.no_grad():
            next_q_values = model(torch.tensor(next_state, dtype=torch.float32))
            target_q_value = car.get_reward() + gamma * torch.max(next_q_values)
        
        # Compute loss and update the model
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(q_values[action], target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        print(f"Episode: {episode}, Episode iteration: {ep_it}, Total Reward: {reward_total}, MSE: {loss}")
        ep_it += 1
    # Decay epsilon after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


 