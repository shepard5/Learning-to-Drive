import numpy as np
from scipy.integrate import quad

def track_proximity_reward(self,car_position, sine_curve):
        safe_margin = .15
        for distance in self.get_sensor_readings:
            if distance<safe_margin:
                reward -= 10
            reward += 1-distance-safe_margin
        return reward  # Squared distance penalty
    
def progress_reward(self):
    def integrand(t):
        return np.sqrt(1+np.sqrt(1+np.cos(t)**2))
    arc_length,_ = quad(integrand,0,self.x)
    reward = arc_length
    return reward

