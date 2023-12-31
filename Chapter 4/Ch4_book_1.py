import gym
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt


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


l1 = 4
l2 = 150
l3 = 150
l4 = 2

# Define the policy network
model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),
    torch.nn.Softmax()
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

MAX_EPISODES = 1000
MAX_DURATION = 500
gamma = 0.99
score = []
losses = []

env = gym.make('CartPole-v0')

for episode in tqdm.tqdm(range(MAX_EPISODES)):
    current_state = env.reset()
    done = False
    # a list of state, action and rewards (we ignore the rewards)
    transitions = []

    for t in range(MAX_DURATION):
        # gets the action probabilities from the model
        action_probabilities = model(torch.from_numpy(current_state).float())
        # samples an action from the distribution
        action = np.random.choice(np.array([0, 1]), p=action_probabilities.data.numpy())

        previous_state = current_state
        current_state, reward, done, info = env.step(action)
        # env.render()
        transitions.append((previous_state, action, t + 1))

        if done:
            break

    # Store the episode length
    ep_len = len(transitions)
    score.append(ep_len)

    # Collect all the rewards in the episode in a single tensor
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))

    # Compute the discounted version of the rewards
    discounted_rewards = discount_rewards(reward_batch)

    # Collect states from a episode in a single tensor

    state_batch = torch.from_numpy(np.array([s for (s, a, r) in transitions])).float()
    action_batch = torch.from_numpy(np.array([a for (s, a, r) in transitions])).float()

    prediction_batch = model(state_batch)
    action_probabilities_batch = prediction_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
    loss = loss_func(action_probabilities_batch, discounted_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())


# test the model after training
test_runs = 5
test_episode_length = []
for i in range(test_runs):
    current_state = env.reset()
    done = False
    test_score = 0
    episode_length = 0
    while not done:
        action_probabilities = model(torch.from_numpy(current_state).float())
        action = np.random.choice(np.array([0, 1]), p=action_probabilities.data.numpy())
        current_state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
        episode_length += 1
    test_episode_length.append(episode_length)

print("Average test score: ", np.mean(test_episode_length))


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


if __name__ == '__main__':
    pass
