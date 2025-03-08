import random
import gymnasium as gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys

num_episodes = 600
batch_size = 100
GAMMA = 0.99
LR = 0.001
TAU = 0.005

EPSILON = 1.0  # Start with full exploration
EPSILON_MIN = 0.01  # Minimum value
EPSILON_DECAY = 0.995  # Decay factor per episode

# if GPU is to be used
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return (
            self.memory
            if batch_size >= self.memory.__len__()
            else random.sample(self.memory, batch_size)
        )

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

replay_memory = ReplayMemory(10000)


def select_action(state):
    if np.random.rand() < EPSILON:
        return torch.tensor(
            [[env.action_space.sample()]], dtype=torch.long, device=device
        )
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Exploit (best action)


# TODO initialize the optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()

for _ in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        # this is where you would insert your policy
        action = select_action(state)

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        next_state, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated

        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        # Store the transition (s,a,r,s′) in the replay buffer.
        replay_memory.push(state, action, next_state, reward, done)

        state = next_state
        if replay_memory.__len__() >= batch_size:
            # Sample a Mini-Batch
            transitions = replay_memory.sample(batch_size=batch_size)
            states, actions, next_states, rewards, dones = zip(*transitions)

            states_batch = torch.cat(states)
            next_states_batch = torch.cat(next_states)
            actions_batch = torch.cat(actions)
            rewards = torch.tensor(rewards, device=device)
            dones = torch.tensor(dones, device=device)

            # Compute Target Q-values
            q_target = (
                GAMMA * target_net(next_states_batch).detach().max(1)[0] * ~dones
                + rewards
            )

            # Compute main Q-values
            q_policy = policy_net(states_batch).gather(1, actions_batch)

            # Compute Huber Loss
            # Huber loss is used to reduce the impact of outliers in the data
            # It is less sensitive to outliers compared to the mean squared error loss
            loss = criterion(q_policy, q_target.unsqueeze(1))

            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            # Backpropagate the loss
            loss.backward()

            # Update the model parameters
            optimizer.step()

        # Update the target network using Soft Updates (Polyak Averaging)
        for target_param, main_param in zip(
            target_net.parameters(), policy_net.parameters()
        ):
            target_param.data.copy_(
                TAU * main_param.data + (1 - TAU) * target_param.data
            )
        # If the episode has ended then we can reset to start a new episode
        if done:
            break
    # Decay epsilon after each episode
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
env.close()
