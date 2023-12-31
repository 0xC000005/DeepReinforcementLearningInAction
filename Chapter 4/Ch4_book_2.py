import gym
import numpy as np
import torch
import tqdm
from collections import deque
import matplotlib.pyplot as plt
import random


def discount_rewards(rewards, gamma=0.99):
    """
    Create a list of exponentially decaying rewards over all actions taken in an episode

    :param rewards: the rewards batch: collection of single rewards for multiple episodes
    :param gamma: the discount factor
    :return: the exponentially decaying rewards
    """
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return


def loss_func(preds, r):
    """
    Compute the loss function
    :param preds:
    :param r:
    :return:
    """
    return -1 * torch.sum(r * torch.log(preds))


l1 = 8
l2 = 300
l3 = 300
l4 = 300
l5 = 4

# Define the policy network
model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),
    torch.nn.ReLU(),
    torch.nn.Linear(l4, l5),
    torch.nn.Softmax()
)

learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

MAX_EPISODES = 1000
MAX_DURATION = 1000
gamma = 1.0001
score = []
losses = []
episode_reward_sum = []

# Parameters
REPLAY_SIZE = 100  # Choose the size of the replay buffer
BATCH_SIZE = 32  # Size of the batch sampled from the replay buffer

# Initialize the replay buffer
replay_buffer = deque(maxlen=REPLAY_SIZE)

env = gym.make('LunarLander-v2')

# Training Loop
for episode in tqdm.tqdm(range(MAX_EPISODES)):
    current_state = env.reset()
    done = False
    episode_rewards = []  # Store rewards for each step in this list
    transitions = []

    for t in range(MAX_DURATION):
        action_probabilities = model(torch.from_numpy(current_state).float())
        action = np.random.choice(np.array([0, 1, 2, 3]), p=action_probabilities.data.numpy())

        previous_state = current_state
        current_state, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        transitions.append((previous_state, action, reward, current_state, done))

        if done:
            break

    # Accumulate total reward of the episode
    total_episode_reward = np.sum(episode_rewards)
    score.append(total_episode_reward)  # Use total reward for scoring

    for transition in transitions:
        replay_buffer.append(transition)

    # Learning from Experience Replay
    if len(replay_buffer) > BATCH_SIZE:
        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(torch.tensor, zip(*minibatch))

        # Prepare for the loss calculation
        state_batch = torch.stack(tuple(state for state in state_batch)).float()
        action_batch = torch.stack(tuple(action for action in action_batch)).long()
        reward_batch = torch.stack(tuple(reward for reward in reward_batch)).float()

        # Compute the action probabilities and then the loss
        prediction_batch = model(state_batch)
        action_probabilities_batch = prediction_batch.gather(dim=1, index=action_batch.view(-1, 1)).squeeze()
        loss = loss_func(action_probabilities_batch, reward_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    episode_reward_sum.append(np.sum(episode_rewards))


# test the model after training
test_runs = 5
test_episode_length = []
test_episode_reward = []
for i in range(test_runs):
    current_state = env.reset()
    done = False
    test_score = 0
    episode_length = 0
    temp_reward = []
    while not done:
        action_probabilities = model(torch.from_numpy(current_state).float())
        action = np.random.choice(np.array([0, 1, 2, 3]), p=action_probabilities.data.numpy())
        current_state, reward, done, info = env.step(action)
        temp_reward.append(reward)
        env.render()
        if done:
            break
        episode_length += 1
    test_episode_length.append(episode_length)
    test_episode_reward.append(np.sum(temp_reward))

print("Average test length: ", np.mean(test_episode_length))
print("Average test reward: ", np.mean(test_episode_reward))


def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0] - N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i + N]
        y[i] /= N
    return y


score = np.array(score)
avg_score = running_mean(score, 50)
plt.figure(figsize=(10, 7))
plt.ylabel("Episode Duration", fontsize=22)
plt.xlabel("Training Epochs", fontsize=22)
plt.plot(avg_score, color='green')
plt.show()

# plot the change of reward sum over time
plt.figure(figsize=(10, 7))
plt.ylabel("Episode Reward", fontsize=22)
plt.xlabel("Training Epochs", fontsize=22)
plt.plot(episode_reward_sum, color='green')
plt.show()



if __name__ == '__main__':
    pass
