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

if __name__ == "__main__": 

    num_episodes = 10000
    learning_rate = 0.0001
    gamma = 0.99  # discount factor
    epsilon_start = 0.9
    epsilon_end = 0.1
    epsilon_decay = .999
    batch_size = 128
    memory_size = 10000
    target_update = 10
    max_timesteps = 100
    input_dim = 9
    output_dim = 5

    car = Car()
    agent = DQN_agent(input_dim,output_dim,learning_rate,gamma)
    memory = ReplayMemory(memory_size)
    epsilon = epsilon_start
    for episode in range (0,num_episodes): 
        
        print(f"{episode}\n")

        car.reset()
        state = car.get_state()
        total_reward = 0
        count = 0

        while True:  

            action = agent.choose_action(state,epsilon)
            car.step(action)
            
            next_state = car.get_state()
            reward, done = car.get_rewards()

            memory.push(state,action,reward,total_reward,next_state,done)
            agent.train(memory)
            agent.update_target_network()

            total_reward += reward
            state = next_state
            count += 1

            if done: 
#               
                print("\n\nNumber of Actions:", count)
                print("\n", car.x, car.y)
                print("\nSingle Run Reward: ", car.get_rewards())
                print(f"\nTotal Reward {total_reward}")
#                print(f"\n{[car.x,car.y]}\n\n")
                break
            
#            print(f"{[car.x,car.y]}")
            epsilon = max(epsilon_end, epsilon*epsilon_decay)

    pulled_experiences = []
