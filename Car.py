import numpy as np
import math
from collections import deque
import random
import numpy as np
from scipy.integrate import quad
from Rewards import track_proximity_reward
from Rewards import progress_reward




class Car: 
    def __init__(self):
        self.x = 0.4   
        self.y = 1.8
        self.velocity = 0
        self.orientation = 0 #0:+y,1:+x,2:-y,3:-x 
        self.scale = 1
        self.action_count = 0
        self.best_runs = deque([(-1000,0,0,0)]*5, maxlen=5)
        self.total_reward = 0
        self.done = False
        pass
    
    def reset(self):
#        self.x = random.uniform(0, 2 * math.pi)
#        self.y = random.uniform(math.sin(self.x) + 1, math.sin(self.x))
        self.x = 0.4   
        self.y = 1.8
        self.velocity = 0
        self.orientation = 0
        self.action_count = 0
        self.total_reward = 0
        self.done = 0
        #self.best_runs = deque([(-1000,0,0,0)]*5, maxlen=5)
        pass
    
    def step(self, action):
        #Coast, Accelerate, Brake, Turn Left, Turn Right 0,1,2,3 respectively
        #Updating attributes based on model's action decision
        #[0:Coast,1:Accelerate,2:Brake,3:Turn Right,4:Turn Left]
        if(action == 0):
            self.velocity = self.velocity*.95
        elif(action == 1) and self.velocity < .45:
            self.velocity += .125
        elif(action == 2) and self.velocity - .25 >= 0:
            self.velocity-= .25
        elif(action == 3):
            self.velocity = self.velocity*.25
            if self.orientation == 0:
                self.orientation = 3
            else:
                self.orientation -= 1
        elif(action == 4):
            self.velocity = self.velocity*.25
            if self.orientation == 3:
                self.orientation = 0
            else:
                self.orientation += 1
        
        # Updating car position
        if self.orientation == 0:
            self.y += self.velocity
        elif self.orientation == 1:
            self.x += self.velocity
        elif self.orientation == 2:
            self.y -= self.velocity
        else:
            self.x -= self.velocity
        
        self.action_count += 1
        

    def check_in_bounds(self): #Returns (in bounds?,finished?)
        if ((math.sin(self.x)+1) <= self.y <= (math.sin(self.x)+2)):
            if self.x > math.tau:
                return True, True
            elif self.x > 0:
                return True, False
        return False, False

            
    def get_sensor_readings(self):
        direction_step = [[0.0,1.0],[1.0,0.0],[0.0,-1.0],[-1.0,0.0]] #Forward for car orientation represented by primary index - direction_step[3] is forward for car orientation = 3 
        sensor_readings = [0,0,0,0]  #Forward,right,backward,left regardless of car orientation
        
        for i in range (0,4):
            temp_car = Car()
            position = np.array([self.x, self.y])  
            temp_car.x = position[0]
            temp_car.y = position[1]
            temp_car_position = np.array([temp_car.x, temp_car.y])
            distance = 0
            
            while True:
                temp_car_position[0] += direction_step[((temp_car.orientation + i) % 4)][0]*self.scale/50
                temp_car_position[1] += direction_step[((temp_car.orientation + i) % 4)][1]*self.scale/50   
                temp_car.x = temp_car_position[0]
                temp_car.y = temp_car_position[1]
                
                result, _ = temp_car.check_in_bounds()
                if result == True:
                     distance += self.scale/50
                else:
                    break
            sensor_readings[i] = distance
        return sensor_readings

    def get_rewards(self): 
        reward = 0

        in_bounds,crossed_finish = self.check_in_bounds() 
        if in_bounds == False and crossed_finish == False:
            self.done = True
            reward -= 50
        elif crossed_finish == True: #Crossed the finish line 
            self.done = True
            reward += 6000
#        reward += self.track_proximity_reward()
        reward += self.progress_reward()

        self.total_reward += reward

        return reward
 
    def get_state(self):
        sensor_readings = self.get_sensor_readings() #length = 4
        combined_state = [self.x,self.y,self.velocity,sensor_readings[0],sensor_readings[1],sensor_readings[2],sensor_readings[3]]
        return combined_state  

    def get_best_runs(self):
        if self.done:
            print("\n\nNumber of Actions:", self.action_count)
            min_tuple = min(self.best_runs, key = lambda x:x[0]) 
            current_tuple = (self.total_reward,self.x,self.y,self.action_count)
      
            if current_tuple[0] > min_tuple[0]:
                self.best_runs.remove(min_tuple)
                self.best_runs.append(current_tuple)
        
        return self.best_runs


    def track_proximity_reward(self):
        reward = 0
        safe_margin = 0.15
        sensor_readings = self.get_sensor_readings()
        
        for distance in sensor_readings:
            if distance < safe_margin:
                reward -= (safe_margin - distance) ** 2  # Penalize for being within the safe margin
            else:
                reward += 0.1  # Small constant reward for being outside the safe margin safely
        
        return reward
        
    def progress_reward(self):
        reward = 0

        def integrand(t):
            return np.sqrt(1+np.cos(t)**2)
        
        arc_length,_ = quad(integrand,0,self.x)
        reward = arc_length
        return reward

