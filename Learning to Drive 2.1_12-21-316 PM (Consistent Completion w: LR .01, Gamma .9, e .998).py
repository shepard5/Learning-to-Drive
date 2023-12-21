# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:56:08 2023

@author: Sam
"""
from Car import Car
from DQN_Agent import DQN_agent
from ReplayMemory import ReplayMemory
import torch.nn as nn 
import torch
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


if __name__ == "__main__": 

    num_episodes = 3000
    learning_rate = 0.001
    gamma = 0.9  # discount factor
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = .998
    batch_size = 100
    memory_size = 15000
    target_update = 10
    max_timesteps = 100
    input_dim = 7
    output_dim = 5

    car = Car()
    agent = DQN_agent(input_dim,output_dim,learning_rate,gamma)
    memory = ReplayMemory(memory_size)
    epsilon = epsilon_start
    best_runs = deque([(-100,0,0)]*5, maxlen=5)
    
    total_rewards_list = deque(maxlen = num_episodes)
    arclength_list = deque(maxlen = num_episodes)

    for episode in range (0,num_episodes): 
        epsilon = max(epsilon_end, epsilon*epsilon_decay)
        
        print(f"{episode}\n")
        
        car.reset()
        state = car.get_state()
        time_step = 0

        while True:  
            time_step += 1

            action = agent.choose_action(state,epsilon)
            car.step(action)
            
            next_state = car.get_state()
            reward = car.get_rewards()
            best_runs = car.get_best_runs()
            total_reward = car.total_reward
            done = car.done

            memory.push(state,action,reward,total_reward,next_state,done)
            agent.train(memory)
            
            if time_step % 10 == 0:
                agent.update_target_network()

            state = next_state
                
            if done:
                for tuple in best_runs:
                    if car.total_reward > tuple[0]:
                        current_tuple = (car.total_reward,car.x,car.y)
                        min_tuple = min(best_runs, key=lambda x: x[0])
                        min_tuple_index = best_runs.index(min(best_runs, key=lambda x: x[0]))
                        best_runs[min_tuple_index] = current_tuple
                
                arc_length = car.progress_reward()
                arc_length_episode = (episode,arc_length)
                arclength_list.append((episode, arc_length))

                episode_total_reward = (episode, total_reward)
                total_rewards_list.append((episode,total_reward))

                print("\n\n",car.x,car.y,total_reward)
                break
        #print(best_runs)


# Display rewards over training
    episode, total_rewards_list = zip(*total_rewards_list)

    episode_array = np.array(episode)
    total_rewards_array = np.array(total_rewards_list)
    slope, intercept = np.polyfit(episode_array, total_rewards_list, 1)
    predicted_y_values = slope * episode_array + intercept

    plt.scatter(episode_array, total_rewards_array)
    

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Coordinates from Deque')

    plt.show()

#Regression of arclength over training

    x_values = np.array([arclength_list[0] for item in arclength_list])
    print(x_values)
    y_values = np.array([arclength_list[1] for item in arclength_list])

    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted_y_values = slope * x_values + intercept
    print("Slope:", slope)
    print("Intercept", intercept)
    
    plt.scatter(x_values,y_values)
    plt.plot(x_values,predicted_y_values, '-r')

    plt.title('Progress Plot')
    plt.show()




