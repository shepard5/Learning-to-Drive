import numpy as np

class Car: 
    def __init__(self):
        self.x = -0.93
        self.y = .03
        self.velocity = 0
        self.orientation = 0 #0:+y,1:+x,2:-y,3:-x 
        self.scale = 1
        self.action_count = 0
        pass
    
    def reset(self):
        self.x = -0.93
        self.y = .03
        self.velocity = 0
        self.orientation = 0
        self.action_count = 0
        pass
    
    def step(self, action):
        #Coast, Accelerate, Brake, Turn Left, Turn Right 0,1,2,3 respectively
        #Updating attributes based on model's action decision
        #[0:Coast,1:Accelerate,2:Brake,3:Turn Right,4:Turn Left]
        if(action == 0):
            pass
        elif(action == 1) and self.velocity < .04:
            self.velocity += .015
        elif(action == 2) and self.velocity - .01 > 0:
            self.velocity-= .01
        elif(action == 3):
            if self.orientation == 0:
                self.orientation = 3
            else:
                self.orientation -= 1
        elif(action == 4):
            if self.orientation == 3:
                self.orientation = 0
            else:
                self.orientation += 1
        
        # Updating car position
        if self.orientation == 0:
            self.x -= self.velocity
        elif self.orientation == 1:
            self.y += self.velocity
        elif self.orientation == 2:
            self.x += self.velocity
        else:
            self.y -= self.velocity
        
        self.action_count += 1
        

    def check_in_bounds(self):
        if (self.x**2 + (self.y) > self.scale) or (self.x**2 + (self.y+.2) < self.scale) or ((self.x < 0) and self.y < 0):
            return False
        else:
            return True
            
    def get_sensor_readings(self):
    # The motivation for this function is to gather information relative to the vehicle, rather than extract information about state as an omnicient observer.
    # The model is learning from the perspective of a car driver.

        #Forward, right, backward, and left relative to the car's orientation.
        direction_step = [[0.0,1.0],[1.0,0.0],[0.0,-1.0],[-1.0,0.0]] #Forward for car orientation represented by primary index - direction_step[3] is forward for car orientation = 3 
        distance_log = [0,0,0,0]  #Forward,right,backward,left regardless of car orientation
        
        for i in range (0,4):
            temp_car = Car()
            position = np.array([self.x, self.y])  
            temp_car.x = position[0]
            temp_car.y = position[1]
            temp_car_position = np.array([temp_car.x, temp_car.y])
            distance = 0
            
            for j in range(0,100):
                temp_car_position[0] += direction_step[((temp_car.orientation + i) % 4)][0]*self.scale/50
                temp_car_position[1] += direction_step[((temp_car.orientation + i) % 4)][1]*self.scale/50   
                temp_car.x = temp_car_position[0]
                temp_car.y = temp_car_position[1]

                if temp_car.check_in_bounds() == True:
                     distance += self.scale/50
                else:
                    break
            distance_log[i] = distance
        sensor_readings = distance_log
        return sensor_readings

    def get_rewards(self): 
        reward = 0.0
        done = False
        if self.check_in_bounds() == False:
            done = True
            reward -= 50
        elif self.y <= 0 and self.x > 0: #Crossed the finish line 
            done = True
            reward += 7500
        if (self.x > 0):
            reward += 5
        if (self.x > 0 and self.y <= 0.6):
            reward += 10
        if self.velocity < .015:
            reward -= 2
        elif self.velocity >= .02:
            reward += 5
        reward += (self.x + .85)*5
        if self.action_count >= 300:
            reward -= float(self.action_count)/10
        return reward, done
 
    def get_state(self):
        sensor_readings = self.get_sensor_readings() #length = 4
        combined_state = [self.x,self.y,self.velocity,self.orientation,self.action_count,sensor_readings[0],sensor_readings[1],sensor_readings[2],sensor_readings[3]]
        return combined_state  