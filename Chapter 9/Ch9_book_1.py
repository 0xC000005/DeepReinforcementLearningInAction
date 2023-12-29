import numpy as np
import torch
import matplotlib.pyplot as plt


def init_grid(size=(10,)):
    grid = torch.randn(*size)
    grid[grid > 0] = 1
    grid[grid <= 0] = 0
    grid = grid.byte()
    return grid


def get_reward(s, a):
    """
    This function takes a neighbors in a s and compares them to agent a;
    if they match, the reward is higher
    :param s:
    :param a:
    :return:
    """
    r = -1
    for i in s:
        if i == a:
            r += 0.9
    r *= 2
    return r


def get_params(N, size):
    ret = []
    for i in range(N):
        vec = torch.randn(size) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret


def qfunc(s, theta, layers=None, afn=torch.tanh):
    """
    This function takes a state s and a set of parameters theta and returns the q value
    :param s: state vector
    :param theta: parameter vector
    :param layers: a list of the form of [(s1, s2), (s3, s4), ...]
    :param afn: activation function
    :return: q value
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
    bias = torch.one((1, theta_1.shape[1]))
    l1 = s @ theta_1 + bias
    l1 = torch.nn.functional.elu(l1)

    # We can also input an activation function to use for the
    # last layer; the default is tanh since rewards ranges from
    # -1 to 1
    l2 = afn(l1 @ theta_2)

    return l2.flatten()


def get_substate(b):
    """
    Takes a single binary number and turns it into a one-hot
    encoded action vector lile [0,1] or [1,0]

    :param b:
    :return:
    """
    s = torch.zeros(2)

    # If the input is 0 (down), the action vector is [1,0]
    if b > 0:
        s[1] = 1
    else:
        s[0] = 1
    return s


def joint_state(s):
    """
    Takes a state vector of two binary numbers and turns it into
    :param s:
    :return:
    """
    s1_ = get_substate(s[0])
    s2_ = get_substate(s[1])
    # Create the joint-action space using the outer-product, then flattens into a vector
    ret = (s1_.reshape(2, 1) @ s2_.reshape(1, 2)).flatten()
    return ret


if __name__ == '__main__':
    size = (20,)
    hid_layer = 20
    params = get_params(size[0], 4 * hid_layer + hid_layer * 2)
    grid = init_grid(size)
    grid_ = grid.clone()
    print(grid)
    plt.imshow(np.expand_dims(grid, axis=0))
    plt.show()

    epochs = 200
    lr = 0.001
    losses = [[] for i in range(size[0])]

    for i in range(epochs):
        for j in range(size[0]):
            l = j - 1 if j -1 >= 0 else size[0] - 1
            r = j + 1 if j + 1 < size[0] else 0

            state_ = grid[[l, r]]
            state = joint_state(state_)
            qvals = qfunc(state.float().detach(),
                          params[j],
                          layers=[(4, hid_layer), (hid_layer, 2)])
            qmax = torch.argmax(qvals, dim=0).detach().item()

            action = int(qmax)
            grid_[j] = action
            reward = get_reward(state_.detach(), action)
            with torch.no_grad():
                target = qvals.clone()
                target[action] = reward
            loss = torch.sum(torch.pow(qvals - target, 2))
                