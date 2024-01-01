import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
from collections import deque
from random import shuffle


def init_grid(size=(10,)):
    grid = torch.randn(*size)
    grid[grid > 0] = 1
    grid[grid <= 0] = 0
    grid = grid.byte()
    return grid


def get_reward(s, a):
    """
    This function takes a neighbors in s and compares them to agent a;
    if they match, the reward is higher
    :param s: all neighbors of the agent
    :param a: the agent
    :return: reward
    """
    r = -1
    for i in s:
        if i == a:
            r += 0.9
    r *= 2
    return r


def get_params(N: int, size: int) -> list:
    """
    This function returns a list of N parameter vectors of size 'size' to be used in the
    qfunc neural network
    :param N:
    :param size:
    :return:
    """
    ret = []
    for i in range(N):
        vec = torch.randn(size) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret


def qfunc(s: list, theta, layers=None, afn=torch.tanh) -> torch.Tensor:
    """
    This function takes a state s and a set of parameter theta and returns the q value
    :param s: state vector, a binary vector of neighbors states
    :param theta: parameter vector
    :param layers: a list of the form of [(s1, s2), (s3, s4), ...]
    :param afn: Activation function
    :return: q value: returns a list of expected rewards for each action (up and down) given the state
    """

    # Takes the first tuple in layers and multiplies those numbers to
    # get the subset of the thera vector to use as the first layer
    if layers is None:
        layers = [(4, 20), (20, 2)]

    lln = layers[0]
    lls = np.prod(lln)

    # Reshape the theta vector subset into a matrix for user as the
    # first layer of the neural network
    theta_1 = theta[:lls].reshape(lln)
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = theta[lls:lls + l2s].reshape(l2n)

    # This is the first layer computation
    # The s input is a joint-action vector of dimension 4,1
    bias = torch.ones((1, theta_1.shape[1]))
    l1 = s @ theta_1 + bias
    l1 = torch.nn.functional.elu(l1)

    # We can also input an activation function to use for the
    # last layer; the default is tanh since rewards ranges from
    # -1 to 1
    l2 = afn(l1 @ theta_2)

    return l2.flatten()


def get_substate(binary):
    """
    Produce state information from the environment (which is the grid)
    Take a single binary number and turns it into a one-hot encoded action
    vector like [1,0] or [0,1]

    :param binary: A bianry number representing the state
    :return:
    """
    action_vector = torch.zeros(2)

    # If the input is 0 (down); the action vector is [1,0]
    if binary > 0:
        action_vector[1] = 1
    else:
        action_vector[0] = 1
    return action_vector


def joint_state(s):
    """
    Get the state vector for each agent
    :param s: s is a vector with two elements where s[0] = left neighbor
    and s[1] = right neighbor
    :return: ret: a joint-action vector of dimension 4,1
    """

    # Gets the action vectors for each element in s
    s1_ = get_substate(s[0])
    s2_ = get_substate(s[1])

    # Create the joint-action space using the outer-product, then flattens into a vector
    ret = (s1_.reshape(2, 1) @ s2_.reshape(1, 2)).flatten()

    return ret


def softmax_policy(qvals, temp=0.9):
    """
    This policy function takes a Q value vector and returns an action, either 0 (down) or 1 (up)
    :param qvals:
    :param temp:
    :return:
    """
    soft = torch.exp(qvals / temp) / torch.sum(torch.exp(qvals / temp))
    action = torch.multinomial(soft, 1)
    return action


def get_coords(grid, j):
    """
    Takes a single index value from the flattened grid and converts it back into [x,y] coordinates
    :param grid:
    :param j:
    :return:
    """
    x = int(np.floor(j / grid.shape[0]))
    y = int(j - x * grid.shape[0])
    return x, y


def get_reward_2d(action, action_mean):
    """
    This is the reward function for the 2D grid
    The reward is based on how different the action is from the mean field action
    :param action:
    :param action_mean:
    :return:
    """
    r = (action * (action_mean - action / 2)).sum() / action.sum()
    return torch.tanh(5 * r)


def get_mean_action(grid, j) -> torch.Tensor:
    """

    :param grid:
    :param j:
    :return:
    """

    # Converts vectorized index j into grid coordinates [x,y], where [0,0] is the top left corner
    x, y = get_coords(grid, j)

    # This will be the action mean vector that we will add to
    action_mean = torch.zeros(2)

    # Two for loops for the 8 neighbors
    for i in [-1, 0, 1]:
        for k in [-1, 0, 1]:
            # This skips the center point
            if i == 0 and k == 0:
                continue

            # This is the neighbor coordinate
            x_, y_ = x + i, y + k

            # This is to handle the edge cases
            x_ = x_ if x_ < grid.shape[0] else grid.shape[0] - 1
            y_ = y_ if y_ < grid.shape[1] else grid.shape[1] - 1
            x_ = x_ if x_ < grid.shape[0] else 0
            y_ = y_ if y_ < grid.shape[1] else 0

            # Convert each neighbor binary spin into an action vector
            cur_n = grid[x_, y_]
            s = get_substate(cur_n)
            action_mean += s

    # Normalize the action mean vector to be a probability distribution
    action_mean /= action_mean.sum()
    return action_mean


if __name__ == '__main__':
    size = (10, 10)
    # TODO: what is J?
    J = np.prod(size)
    hid_layer = 10
    layers = [(2, hid_layer), (hid_layer, 2)]
    params = get_params(N=1, size=2 * hid_layer + hid_layer * 2)
    grid = init_grid(size=size)
    grid_ = grid.clone()
    grid__ = grid.clone()

    plt.imshow(grid)
    plt.show()
    print(grid.sum())

    epochs = 75
    lr = 0.0001
    num_iter = 3  # A
    losses = [[] for i in range(size[0])]  # B
    replay_size = 50  # C
    replay = deque(maxlen=replay_size)  # D
    batch_size = 10  # E
    gamma = 0.9  # F
    losses = [[] for i in range(J)]

    for i in tqdm.tqdm(range(epochs)):
        act_means = torch.zeros((J, 2))  # G
        q_next = torch.zeros(J)  # H
        for m in range(num_iter):  # I
            for j in range(J):  # J
                action_mean = get_mean_action(grid_, j).detach()
                act_means[j] = action_mean.clone()
                qvals = qfunc(action_mean.detach(), params[0], layers=layers)
                action = softmax_policy(qvals.detach(), temp=0.5)
                grid__[get_coords(grid_, j)] = action
                q_next[j] = torch.max(qvals).detach()
            grid_.data = grid__.data
        grid.data = grid_.data
        actions = torch.stack([get_substate(a.item()) for a in grid.flatten()])
        rewards = torch.stack([get_reward_2d(actions[j], act_means[j]) for j in range(J)])
        exp = (actions, rewards, act_means, q_next)  # K
        replay.append(exp)
        shuffle(replay)
        if len(replay) > batch_size:  # L
            ids = np.random.randint(low=0, high=len(replay), size=batch_size)  # M
            exps = [replay[idx] for idx in ids]
            for j in range(J):
                jacts = torch.stack([ex[0][j] for ex in exps]).detach()
                jrewards = torch.stack([ex[1][j] for ex in exps]).detach()
                jmeans = torch.stack([ex[2][j] for ex in exps]).detach()
                vs = torch.stack([ex[3][j] for ex in exps]).detach()
                qvals = torch.stack([qfunc(jmeans[h].detach(), params[0], layers=layers) \
                                     for h in range(batch_size)])
                target = qvals.clone().detach()
                target[:, torch.argmax(jacts, dim=1)] = jrewards + gamma * vs
                loss = torch.sum(torch.pow(qvals - target.detach(), 2))
                losses[j].append(loss.item())
                loss.backward()
                with torch.no_grad():
                    params[0] = params[0] - lr * params[0].grad
                params[0].requires_grad = True

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.array(losses).mean(axis=0))
    ax[1].imshow(grid)
    plt.show()
