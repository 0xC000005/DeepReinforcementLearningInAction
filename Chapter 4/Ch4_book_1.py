import gym
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
state1 = env.reset()

l1 = 4
l2 = 150
l3 = 2

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax()
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pred = model(torch.from_numpy(state1).float())
action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
state2, reward, done, info = env.step(action)
env.render()


def discount_rewards(rewards, gamma=0.99):
    # Compute the exponentially decaying rewards
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return


def loss_func(preds, r):
    return -1 * torch.sum(r * torch.log(preds))


MAX_DUR = 200
MAX_EPISODES = 500
gamma = 0.99
score = []
losses = []
for episode in tqdm.tqdm(range(MAX_EPISODES)):
    current_state = env.reset()
    done = False
    transitions = []

    for t in range(MAX_DUR):
        action_probabilities = model(torch.from_numpy(current_state).float())
        action = np.random.choice(np.array([0, 1]), p=action_probabilities.data.numpy())

        previous_state = current_state
        current_state, reward, done, info = env.step(action)
        # env.render()
        transitions.append((previous_state, action, reward))

        if done:
            break

        # Store the episode length
        ep_len = len(transitions)
        score.append(ep_len)
        # Collect all the rewards in the episode in a single tensor
        reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
        # Compute the discounted version of the rewards
        discounted_rewards = discount_rewards(reward_batch)
        state_batch = torch.Tensor([s for (s, a, r) in transitions])
        action_batch = torch.Tensor([a for (s, a, r) in transitions])
        prediction_batch = model(state_batch)
        action_probabilities_batch = prediction_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()
        loss = loss_func(action_probabilities_batch, discounted_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())


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
plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_score, color='green')
plt.show()

if __name__ == '__main__':
    pass
