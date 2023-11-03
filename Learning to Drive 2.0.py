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

class Track:
    def __init__(self, rows=100, cols=100, width = 6):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.track = self.oval_setup_track()
        
    def oval_setup_track(self):
        # Define track boundaries and features here
        # For example, a simple oval track
        track = np.zeros((self.rows, self.cols))
        
        #Center of ellipse
        center_x, center_y = self.cols //2, self.rows // 2
        
        # Radii of the ellipse
        radius_x_outer, radius_y_outer = self.cols // 2 - 1, self.rows // 2 - 1
        radius_x_inner, radius_y_inner = self.cols // 2 - self.width, self.rows // 2 - self.width
        
        # Create a grid of points
        y, x = np.ogrid[:self.rows, :self.cols]
        
        #Define inner and outer ovals as boolean masks
        outer_oval = ((x - center_x)**2 / radius_x_outer**2) + ((y - center_y)**2 / radius_y_outer**2) <= 1
        inner_oval = ((x - center_x)**2 / radius_x_inner**2) + ((y - center_y)**2 / radius_y_inner**2) <= 1
        
        
        #Define the track by the area between the outer and inner ovals
        track[outer_oval] = 1
        track[inner_oval] = 0
        
        # Add start and finish lines
        start_finish_width = 3
        start_line_y = center_y - radius_y_outer
        start_line_x = center_x 
        end_line_y = start_line_y + start_finish_width
        
        print(f"start y values {start_line_y + 3} , start x value {center_x - 2}")
        
        #Start Line assigned values of 2
        track[start_line_y+1:self.width+1, start_line_x-start_finish_width:start_line_x] = 2
        
        #Finish Line assigned values of 3
        track[start_line_y+1:self.width+1, start_line_x:start_line_x+start_finish_width] = 3
        
        return track
    
    def display(self):
        plt.imshow(self.track, origin="lower", cmap="gray_r")
        plt.title("Track Visualization")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    def check_position(self, x, y):
        if track[y,x] == 0:
            car_has_crashed = True
        ## Continue logic if necessary
        pass

class Car:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.velocity = 0
        self.orientation = 0
        pass
    
    def reset(self):
        self.x = 0
        self.y = 0
        self.velocity = 0
        self.orientation = 0
        pass
    
    
    def step(self, action):
        
        #Accelerate, Brake, Turn Left, Turn Right 0,1,2,3 respectively
        #Updating attributes based on model's action decision
        
        if(action == 0):
            self.velocity += 1
        elif(action == 1):
            self.velocit -= 1
        elif(action == 2):
            if self.orientation == 0:
                self.orientation = 3
            else:
                self.orientation -= 1
        elif(action == 3):
            if self.orientation == 3 :
                self.orientation = 0
            else:
                self.orientation += 1
        
        # Updating car position
        if self.orientation == 0:
            self.x -= velocity
        if self.orientation == 1:
            self.y += velocity
        if self.orientation == 2:
            self.x += velocity
        if self.orientation == 3:
            self.y -= velocity
        pass

    def get_sensor_readings(self, track):
        """
        Get sensor readings for distance to track boundaries in four directions:
        forward, backward, left, and right relative to the car's orientation.
        """

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

            # Move in the direction until we hit the track boundary (value 1) or another obstacle
            #while True:
                #position += direction
                #if position[0] < 0 or position[0] >= track.shape[1] or \
                   #position[1] < 0 or position[1] >= track.shape[0] or \
                   #track[int(position[1]), int(position[0])] == 1:
                    #break
                #distance += 1
            
            while True:
                position += direction
                if position[0] < 0 or position[0] >= track.shape[1] or \
                   position[1] < 0 or position[1] >= track.shape[0] or \
                   track[int(position[1]), int(position[0])] == 1:
                    break
                distance += 1
            sensor_readings[key] = distance

        return sensor_readings




class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define network structure
        pass
    
    def forward(self, x):
        # Define forward pass
        pass



track = Track()
track.display()