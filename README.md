# Neural-Net-Playground
Just playing around with different NN libraries and exploring their capabilities

Horse Racing... Initial assumptions for bulding the model: Individual finishing times are independent, horse finishes are dependent, (are stable positions independent of horse times?)
creating a model that predicts finishing times will be simpler because of independence.

"Learning to Drive" and "Learning to Drive 2.0" is a DQN - trains the model approximating q-values (most optimal next step based on prior info) using the Bellman equation. At each step, car can brake, turn left, right and accel (4 actions). 

Image classification playground templates a convolution NN for any image related datasets - (image formatting in script). 91% cancer identification success rate using pre-optimized hyperparameters from https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym  # or your custom environment

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Define your network architecture here
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)

# Hyperparameters
num_episodes = 1000
learning_rate = 0.001
gamma = 0.99  # discount factor
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay = 200
batch_size = 64
memory_size = 10000
target_update = 10

# Environment setup
env = gym.make('CartPole-v1')  # Replace with your environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# DQN and optimizer
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Replay memory
memory = deque(maxlen=memory_size)

# Training loop
for episode in range(num_episodes):
    state = torch.tensor([env.reset()], dtype=torch.float32)
    for t in range(1000):  # Replace 1000 with max timesteps in your environment
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1. * episode / epsilon_decay)
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)

        # Store the transition in memory
        memory.append((state, action, next_state, reward, done))

        # Move to the next state
        state = next_state

        # Perform one step of optimization on the policy network
        if len(memory) > batch_size:
            transitions = random.sample(memory, batch_size)
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)

            batch_state = torch.cat(batch_state)
            batch_action = torch.cat(batch_action)
            batch_next_state = torch.cat(batch_next_state)
            batch_reward = torch.cat(batch_reward)
            batch_done = torch.tensor(batch_done, dtype=torch.uint8)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            current_q_values = policy_net(batch_state).gather(1, batch_action)

            # Compute V(s_{t+1}) for all next states.
            next_q_values = target_net(batch_next_state).max(1)[0].detach()
            expected_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))

            # Compute Huber loss
            loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Update the target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Optional: Print episode results here

env.close()
