import random
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt


class ContextualBandit:
    def __init__(self, number_of_bandits=3):
        self.number_of_bandits = number_of_bandits
        self.number_of_states = number_of_bandits
        self.bandit_matrix = None
        self.initialize_state_reward_distribution_table()
        self.current_state = None
        self.update_state()

    def initialize_state_reward_distribution_table(self):
        # self.bandit_matrix = np.random.rand(self.number_of_states, self.number_of_bandits)
        self.bandit_matrix = np.array([[0.0, 0.0, 0.7], [0.0, 0.7, 0.0], [0.7, 0.0, 0.0]])
    def reward(self, probability):
        temp_reward = 0
        for i in range(10):
            if random.random() < probability:
                temp_reward += 1
        return temp_reward

    def get_current_state(self):
        return self.current_state

    def update_state(self):
        self.current_state = np.random.randint(0, self.number_of_states)

    def get_reward(self, bandit):
        probability = self.bandit_matrix[self.get_current_state()][bandit]
        reward = self.reward(probability)
        return reward

    def choose_bandit(self, bandit):
        reward = self.get_reward(bandit)
        self.update_state()
        return reward


NUMBER_OF_BANDITS = 3


def one_hot_encoding(bandit, value=1):
    one_hot_vector = np.zeros(NUMBER_OF_BANDITS)
    one_hot_vector[bandit] = value
    return one_hot_vector


def softmax(vector, tau=2.12):
    total = sum([np.exp(vector[i] / tau) for i in range(len(vector))])
    softmax_probability = [np.exp(vector[i] / tau) / total for i in range(len(vector))]
    return softmax_probability


def train(env, epochs=5000, learning_rate=1e-2):
    current_state = torch.Tensor(one_hot_encoding(env.get_current_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    running_average_of_most_recent_10_rewards_received_over_time = [0]
    most_recent_10_rewards_received = np.zeros(10)
    for i in tqdm.tqdm(range(epochs)):
        reward_prediction = model(current_state)
        # convert reward prediction to probability distribution with softmax
        reward_probability_distribution = softmax(reward_prediction.data.numpy(), tau=2.12)
        # normalize probability distribution
        reward_probability_distribution /= sum(reward_probability_distribution)
        chosen_bandit = np.random.choice(NUMBER_OF_BANDITS, p=reward_probability_distribution)
        current_reward = env.choose_bandit(chosen_bandit)

        # update the most recent 10 rewards received: pop the oldest reward and append the newest reward
        most_recent_10_rewards_received = np.append(most_recent_10_rewards_received[1:], current_reward)
        # update the running average of the most recent 10 rewards received
        running_average_of_most_recent_10_rewards_received_over_time.append(
            np.mean(most_recent_10_rewards_received))

        # convert Pytorch tensor to numpy array
        one_hot_encoded_reward = reward_prediction.data.numpy().copy()
        # update the chosen bandit with the current reward for training
        one_hot_encoded_reward[chosen_bandit] = current_reward
        current_reward = torch.Tensor(one_hot_encoded_reward)
        rewards.append(current_reward)

        # calculate the loss
        loss = loss_function(reward_prediction, current_reward)
        # zero the gradients before backward pass because Pytorch accumulates the gradients
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update the weights
        optimizer.step()
        # update the current environment state
        current_state = torch.Tensor(one_hot_encoding(env.get_current_state()))

    return np.array(rewards), running_average_of_most_recent_10_rewards_received_over_time


if __name__ == '__main__':
    N, D_in, H, D_out = 1, NUMBER_OF_BANDITS, 100, NUMBER_OF_BANDITS

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    loss_function = torch.nn.MSELoss(reduction='sum')

    env = ContextualBandit(NUMBER_OF_BANDITS)

    rewards, running_average_of_most_recent_10_rewards_received_over_time = train(env)

    plt.plot(running_average_of_most_recent_10_rewards_received_over_time)
    plt.xlabel('Plays')
    plt.ylabel('Avg Reward')
    plt.title('Average Reward vs Plays')
    plt.show()

    for i in rewards:
        print(i)



