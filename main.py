import random
import gymnasium as gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

num_episodes = 600
batch_size = 100
GAMMA = 0.99
LR = 0.001

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
        return random.sample(self.memory, batch_size)


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

# TODO initialize the optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR)


for _ in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while True:
        # this is where you would insert your policy
        # TODO select the action
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Store the transition (s,a,r,s′) in the replay buffer.
        replay_memory.push(state, action, next_state, reward, done)

        # Sample a Mini-Batch
        mini_batch = replay_memory.sample(batch_size=batch_size)

        states, actions, next_states, rewards, dones = zip(*mini_batch)

        # Compute Target Q-values
        q_target = rewards + GAMMA * target_net(next_states).detach().max(1)[0] * (
            1 - dones
        )

        # Compute main Q-values
        q_policy = policy_net(states)

        # Compute Huber Loss
        # Huber loss is used to reduce the impact of outliers in the data
        # It is less sensitive to outliers compared to the mean squared error loss
        loss = F.smooth_l1_loss(q_policy, q_target)

        # Zero the gradients of the optimizer
        optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # If the episode has ended then we can reset to start a new episode
        if done:
            break

env.close()
