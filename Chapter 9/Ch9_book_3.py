import magent
import math
from scipy.spatial.distance import cityblock
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
    :param s: all neighbors of
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


map_size = 30
env = magent.GridWorld("battle", map_size=map_size)
team1, team2 = env.get_handles()

hid_layer = 25
in_size = 359
act_space = 21
layers = [(in_size, hid_layer), (hid_layer, act_space)]
params = get_params(2, in_size * hid_layer + hid_layer * act_space)  # A
map_size = 30
width = height = map_size
n1 = n2 = 16  # B
gap = 1  # C
epochs = 100
replay_size = 70
batch_size = 25

side1 = int(math.sqrt(n1)) * 2
pos1 = []
for x in range(width // 2 - gap - side1, width // 2 - gap - side1 + side1, 2):  # D
    for y in range((height - side1) // 2, (height - side1) // 2 + side1, 2):
        pos1.append([x, y, 0])

side2 = int(math.sqrt(n2)) * 2
pos2 = []
for x in range(width // 2 + gap, width // 2 + gap + side2, 2):  # E
    for y in range((height - side2) // 2, (height - side2) // 2 + side2, 2):
        pos2.append([x, y, 0])

env.reset()
env.add_agents(team1, method="custom", pos=pos1)  # F
env.add_agents(team2, method="custom", pos=pos2)

if __name__ == '__main__':
    pass
