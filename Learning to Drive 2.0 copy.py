# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:56:08 2023

@author: Sam
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
import math

class Track:
    def __init__(self, radius, width):
        self.radius = radius
        self.width = width #Proportional Scaling for the inner border of the track (.9 would produce a circle 90 percent the size of the outer circle)
        self.track = self.setup_track()
        
    def track_coordinates(self,t):
        # Define track boundaries and features here
        # For example, a simple oval track
        track = [x_inner,y_inner,x_outer,y_outer]

        if t < math.pi:
            x_inner = self.radius + self.radius*math.cos(t)
            y_inner = self.radius + self.radius*math.sin(t)

            x_outer = self.width * (self.radius + self.radius*math.cos(t))
            y_outer = self.width * (self.radius + self.radius*math.sin(t))
        else:
            x_inner = -self.radius + self.radius*math.cos(t)
            y_inner = -self.radius + self.radius*math.sin(t)

            x_outer = self.width * (self.radius + self.radius*math.cos(t))
            y_outer = self.width * (self.radius + self.radius*math.sin(t))
        
        
        return track
    
    def display(self):
        plt.imshow(self.track, origin="lower", cmap="gray_r")
        plt.title("Track Visualization")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    

## May need an extra conditional statement if positions are exactly on the bisectional x and y axes ex: (1/2*radius,y*) or (x*,1/2*radius)

class Car:
    def __init__(self):
        self.x = .95
        self.y = .5
        self.velocity = 0
        self.orientation = 1
        self.crash_eval = 0
        pass
    
    def reset(self):
        self.x = .95
        self.y = .5
        self.velocity = 0
        self.orientation = 1
        pass
    
    
    def step(self, action):
        
        #Coast, Accelerate, Brake, Turn Left, Turn Right 0,1,2,3 respectively
        #Updating attributes based on model's action decision
        
        if(action == 0):
            pass
        elif(action == 1) & self.velocity < .04:
            self.velocity += .015
        elif(action == 2) & self.velocity > 0:
            self.velocity-= .01
        elif(action == 3):
            if self.orientation == 0:
                self.orientation = 3
            else:
                self.orientation -= 1
        elif(action == 4):
            if self.orientation == 3 :
                self.orientation = 0
            else:
                self.orientation += 1
        
        # Updating car position
        if self.orientation == 0:
            self.x -= self.velocity
        if self.orientation == 1:
            self.y += self.velocity
        if self.orientation == 2:
            self.x += self.velocity
        if self.orientation == 3:
            self.y -= self.velocity
        pass

    def get_sensor_readings(self, track):

        #forward, backward, left, and right relative to the car's orientation.

        # Maps of orientation to direction vectors
        directions = {
            0: {'forward': np.array([0, 1]), 'backward': np.array([0, -1]), 'left': np.array([-1, 0]), 'right': np.array([1, 0])},
            1: {'forward': np.array([1, 0]), 'backward': np.array([-1, 0]), 'left': np.array([0, 1]), 'right': np.array([0, -1])},
            2: {'forward': np.array([0, -1]), 'backward': np.array([0, 1]), 'left': np.array([1, 0]), 'right': np.array([-1, 0])},
            3: {'forward': np.array([-1, 0]), 'backward': np.array([1, 0]), 'left': np.array([0, -1]), 'right': np.array([0, 1])},
        }

        sensor_directions = directions[self.orientation]

        sensor_readings = {}

        # Check each sensor direction
        for key, direction in sensor_directions.items():
            position = np.array([self.x, self.y])  # Car's position
            distance = 0
            
            while True:
                position += direction
                if position[0] < 0 or position[0] >= track.shape[1] or \
                   position[1] < 0 or position[1] >= track.shape[0] or \
                   track[0](self.x) == 1:
                    break
                distance += 1
            sensor_readings[key] = distance

        return sensor_readings
    
    def check_crash(self, track):

        ## Need to evaluate values of x,y for each and check they're in range... given car x, calculate both curve's y ... if not in range then crashed. ex: 
        #car position : x = .93, y = .52 : x_inner = .93 , y_inner = ? solve for t, .93 = = self.radius + self.radius*math.cos(t) so invcos((.93 - self.radius)/self.radius) = t ... 
        #proceed to calculate t to evaluate y. Repaear for the outer parametric curve ie outer circle and put an if statement to check for crash
        
        crash_eval = False

        t_inner = math.acos((self.x-track.radius)*track.radius**-1)
        y_check_inner = y_inner = track.radius + track.radius*math.sin(t_inner)

        t_outer = math.acos((self.x-track.radius)*track.radius**-1)
        y_check_outer = track.width * (track.radius + track.radius*math.sin(t_outer))




        if self.y > track.radius*1/2 & self.x > track.radius*1/2: 
            if (self.y > y_check_inner) & (self.y < y_check_outer):
                pass
            else: 
                crash_eval = True
        elif self.y > track.radius*1/2 & self.x < track.radius*1/2: 
            if (self.y > y_check_inner) & (self.y < y_check_outer):
                pass
            else:
                crash_eval = True
        elif self.y < track.radius*1/2 & self.x < track.radius*1/2: 
            if (self.y < y_check_inner) & (self.y > y_check_outer):
                pass
            else: 
                crash_eval = True
        elif self.y < track.radius*1/2 & self.x > track.radius*1/2: 
            if (self.y < y_check_inner) & (self.y > y_check_outer):
                pass
            else: 
                crash_eval = True
        return crash_eval

        if self.y > track.radius*1/2 & self.x > track.radius*1/2: 
            if self.x > track[0] & self.x < track[2] & self.y > track[1] * self.y < track[3]:
                pass
            else: 
                crash_eval = True
        elif self.y > track.radius*1/2 & self.x < track.radius*1/2: 
            if self.x < track[0] & self.x > track[2] & self.y > track[1] * self.y < track[3]:
                pass
            else:
                crash_eval = True
        elif self.y < track.radius*1/2 & self.x < track.radius*1/2: 
            if self.x < track[0] & self.x > track[2] & self.y < track[1] * self.y > track[3]:
                pass
            else: 
                crash_eval = True
        elif self.y < track.radius*1/2 & self.x > track.radius*1/2: 
            if self.x < track[0] & self.x > track[2] & self.y < track[1] * self.y > track[3]:
                pass
            else: 
                crash_eval = True
        return crash_eval
    
    def get_rewards(self, track): 
        reward = 0
        if self.check_crash(track):
            return -100
        elif self.has_finished: 
            return 5000
        if self.veliocity < 5:
            reward -= 5
        elif self.velocity > 10:
            reward += 10
        return reward
    
    def get_state(self,track):
        sensor_readings = self.get_sensor_readings(self,track) #length = 4
        return self.x, self.y, self.velocity, self.orientation, self.crash_eval, sensor_readings



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

##Training
def choose_action(state, model, epsilon): 
    if random.uniform(0,1) < epsilon:
        return random.choice(range(model.fc3.out_features)) #10% probability a random action is selected (exploration)
    else:
        state_tensor = torch.tensor(state,dtype = torch.float32).unsqueeze(0) #90% probability max Q action is selected (exploitation)
        with torch.no_grad():
            q_values = model(state_tensor)
        return torch.argmax(1_values).item

num_episodes = 1000
learning_rate = 0.001
gamma = 0.99  # discount factor
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 200
batch_size = 64
memory_size = 10000
target_update = 10
max_timesteps = 50

car = Car()
track = Track()

memory = deque(maxlen=memory_size)

for episode in range (num_episodes):
    state = car.reset()
    for t in range(max_timesteps):
        action = choose_action()