import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))

        # The actor head returns the log probabilities over the 2 actions
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))

        # The critic returns a single number bounded by (-1, 1)
        critic = torch.tanh(self.critic_lin1(c))

        # Returns the actor and critic result as a tuple
        return actor, critic


def run_episode(worker_env, worker_model):
    # Converts the environment state from a numpy array to a PyTorch tensor
    state = torch.from_numpy(worker_env.env.state).float()
    # Creates lists to store the computed state values (critic), log probabilities (actor), and rewards
    values, log_probabilities, rewards = [], [], []

    done = False
    j = 0

    # Keeps playing the game until the episode ends
    while not done:  # C
        j += 1

        # Computes the state value and log probabilities over actions
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)

        # Using the actor's log probabilities over actions, creates and samples from a categorical
        # distribution to get an action
        action = action_dist.sample()

        log_probabilities_ = policy.view(-1)[action]
        log_probabilities.append(log_probabilities_)

        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()

        # If the last action caused the episode to end, sets the reward to -10 and
        # resets the environment
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0

        rewards.append(reward)
    return values, log_probabilities, rewards


def update_params(worker_opt, values, log_probabilities, rewards, clc=0.1, gamma=0.95):
    # We reverse the order of the rewards, log probabilities, and value_ arrays
    # and call .view(-1) to make sure they are flat
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    log_probabilities = torch.stack(log_probabilities).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    returns = []
    ret_ = torch.Tensor([0])

    # For each reward (in reverse order), we compute the return value and append it to a returns array
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        returns.append(ret_)

    returns = torch.stack(returns).view(-1)
    returns = F.normalize(returns, dim=0)

    # We need to detach the values tensor from the graph to prevent backpropagation through the critic head
    actor_loss = -1 * log_probabilities * (returns - values.detach())

    # The critic attempts to learn to predict the return
    critic_loss = torch.pow(values - returns, 2)

    # We sum the actor and critic losses to get an overall loss.
    # We scale down then critic loss by the clc factor
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()

    worker_opt.step()

    return actor_loss, critic_loss, len(rewards)


def worker(t, worker_model, counter, params, all_episode_lengths):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    # Each process runs its own isolate environment and optimizer but shares the model
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()
    episode_lengths = []
    for i in tqdm(range(params['epochs'])):
        worker_opt.zero_grad()

        # The run_episode function plays an episode of the game, collecting data along the way
        values, log_probabilities, rewards = run_episode(worker_env, worker_model)

        # we use the collected data from run_episode to run one parameter update step
        actor_loss, critic_loss, episode_length = update_params(worker_opt, values, log_probabilities, rewards)

        # Counter is a globally shared counter between all the running processes
        counter.value = counter.value + 1

        # We append the episode length to a list to track the progress of the algorithm
        episode_lengths.append(episode_length)

        # Append the episode lengths to the shared list instead of returning it
    all_episode_lengths.append(episode_lengths)


if __name__ == '__main__':
    # Creates a global, shared actor-critic model
    MasterNode = ActorCritic()

    # The shared_memory() method will allow the parameters of the model to the shared across processes rather than
    # being copied.
    MasterNode.share_memory()

    # Sets up a list to store the instantiated processes
    processes = []

    # Initialize a multiprocessing Manager and a Manager list to collect episode lengths from all workers
    manager = mp.Manager()
    all_episode_lengths = manager.list()

    params = {
        'epochs': 1000,
        'n_workers': 40,
    }

    # A shared global counter using multiprocessing built-in shared objects.
    # The 'i' parameter indicates the type is integer
    counter = mp.Value('i', 0)

    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params, all_episode_lengths))
        p.start()
        processes.append(p)

    # "Joins" each process to wait for it to finish before returning to the main process
    for p in processes:
        p.join()

    # Makes sure each process is terminated
    for p in processes:
        p.terminate()

    # Prints the global counter-value and the first process's exit code (0)
    print(counter.value, processes[0].exitcode)

    all_episode_lengths = list(all_episode_lengths)

    # plot the smooth episode lengths for each worker over time
    # calculate a moving average (nd array) over 10 episodes for each worker's episode lengths
    smoothed_episode_lengths = [np.convolve(ep, np.ones((10,)) / 10, mode='valid') for ep in all_episode_lengths]

    # plot the nd arrays episode lengths for each worker
    for ep in smoothed_episode_lengths:
        plt.plot(ep)
    plt.xlabel('Epoch')
    plt.ylabel('Episode Length')

    # save the plot
    plt.savefig('episode_lengths.png')

    # Save the model
    torch.save(MasterNode.state_dict(), 'model.pt')

    # env = gym.make("CartPole-v1")
    # env.reset()
    #
    # for i in range(1000):
    #     state_ = np.array(env.env.state)
    #     state = torch.from_numpy(state_).float()
    #     logits, value = MasterNode(state)
    #     action_dist = torch.distributions.Categorical(logits=logits)
    #     action = action_dist.sample()
    #     state2, reward, done, info = env.step(action.detach().numpy())
    #     if done:
    #         print("Lost")
    #         env.reset()
    #     state_ = np.array(env.env.state)
    #     state = torch.from_numpy(state_).float()
    #     env.render()
