import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import copy

from collections import deque, namedtuple


# Define the DQN architecture
class DQN_dynamic(nn.Module):
    def __init__(self):
        super(DQN_dynamic, self).__init__()
        self.fc1 = nn.Linear(8, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)
        self.fc4 = nn.Linear(24, 3)

    def forward(self, x, restricted):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if restricted:
            return self.fc3(x)
        else:
            return self.fc3(x), self.fc4(x)


# Define the Double DQN logic
class DoubleDQN_dynamic:
    def __init__(self):
        self.dqn = DQN_dynamic()
        self.target_dqn = DQN_dynamic()
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.optimizer = optim.Adam(self.dqn.parameters())
        self.criterion = nn.MSELoss()

        self.replay_buffer_unrestricted = []
        self.replay_buffer_restricted = []
        self.buffer_capacity = 10000

    def push_to_buffer(self, exp, restricted):  # redo as deque
        state, action, reward, next_state, done = exp
        if restricted:
            self.replay_buffer_restricted.append((state, action, reward, next_state, done))
            if len(self.replay_buffer_restricted) > self.buffer_capacity:
                del self.replay_buffer_restricted[0]
        if not restricted:
            self.replay_buffer_unrestricted.append((state, action, reward, next_state, done))
            if len(self.replay_buffer_unrestricted) > self.buffer_capacity:
                del self.replay_buffer_unrestricted[0]

    def sample_from_buffer(self, batch_size, restricted):
        if restricted:
            return random.sample(self.replay_buffer_restricted, batch_size)
        if not restricted:
            return random.sample(self.replay_buffer_unrestricted, batch_size)

    def take_action(self, state, restricted=True, epsilon=0.5, action_mask=None):
        if restricted:
            if np.random.rand() < epsilon:
                random_action_probs = abs(np.random.randn(3))
                masked_action_probabilities = random_action_probs * action_mask
                action = np.argmax(masked_action_probabilities)
                return action
            else:
                with torch.no_grad():
                    q1 = self.dqn(state, restricted)
                    # TODO: what does the action mask param means?
                    action_mask = torch.FloatTensor(action_mask)
                    masked_q_values = q1 + (action_mask * 1e6 - 1e6)
                    action1 = torch.argmax(masked_q_values)
                    return action1.item()
        if not restricted:
            q1, q2 = self.dqn(torch.FloatTensor(state), restricted)
            action1 = torch.argmax(q1)
            action2 = torch.argmax(q2)
            action1 = action1.item()
            action2 = action2.item()
            if np.random.rand() < epsilon:
                action1 = np.random.choice(3)
            if np.random.rand() < epsilon:
                action2 = np.random.choice(3)
            return action1, action2

    def update_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train(self, restricted=False, batch_size=32, gamma=0.0):  # Set gamma to 0
        if restricted:
            replay_buffer = self.replay_buffer_restricted
        if not restricted:
            replay_buffer = self.replay_buffer_unrestricted

        if len(replay_buffer) < batch_size:
            return

        batch = self.sample_from_buffer(batch_size, restricted)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)  # Stack the tensors
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)  # Stack the tensors
        dones = torch.FloatTensor(dones)

        if restricted:
            q_values = (
                self.dqn(states, restricted).gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            target = rewards  # Directly set target to rewards as gamma is 0
            loss = self.criterion(q_values, target)
        if not restricted:
            q_values1, q_values2 = self.dqn(states, restricted)
            q_values1 = q_values1.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_values2 = q_values2.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss1 = self.criterion(q_values1, rewards)
            loss2 = self.criterion(q_values2, rewards)
            loss = loss1 + loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_td(self, restricted=False, batch_size=32, gamma=0.99):  # Adjust gamma as needed
        # this is not tested yet
        if restricted:
            replay_buffer = self.replay_buffer_restricted
        if not restricted:
            replay_buffer = self.replay_buffer_unrestricted

        if len(replay_buffer) < batch_size:
            return

        batch = self.sample_from_buffer(batch_size, restricted)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)

        if restricted:
            current_q_values = self.dqn(states, restricted).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_dqn(next_states, restricted).detach().max(1)[0]
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)
            loss = self.criterion(current_q_values, expected_q_values)
        else:
            current_q_values1, current_q_values2 = self.dqn(states, restricted)
            current_q_values1 = current_q_values1.gather(1, actions.unsqueeze(1)).squeeze(1)
            current_q_values2 = current_q_values2.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Use target network to get the next Q values
            next_q_values1, next_q_values2 = self.target_dqn(next_states, restricted)
            next_q_values = torch.min(next_q_values1.detach().max(1)[0], next_q_values2.detach().max(1)[0])

            expected_q_values = rewards + gamma * next_q_values * (1 - dones)
            loss1 = self.criterion(current_q_values1, expected_q_values)
            loss2 = self.criterion(current_q_values2, expected_q_values)
            loss = loss1 + loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
