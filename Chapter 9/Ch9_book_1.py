import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm


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


if __name__ == '__main__':
    plt.figure(figsize=(8, 5))
    size = (20,)
    hid_layer = 20
    # Generates a list of parameter vectors that will parameterize the Q functions used for each agent
    params = get_params(size[0], 4 * hid_layer + hid_layer * 2)
    grid = init_grid(size)
    grid_ = grid.clone()
    print("init grid: ", grid)
    plt.imshow(np.expand_dims(grid, axis=0))

    epochs = 200
    learning_rate = 0.001  # A

    # Since we are dealing with multiple agents, each controlled by a separate Q function,
    # we have to keep track of multiple losses
    losses = [[] for i in range(size[0])]  # B
    for i in tqdm.tqdm(range(epochs)):
        # Iterates through each agent
        for j in range(size[0]):
            # Gets the left neighbor; if at the beginning, loops to the end
            left = j - 1 if j - 1 >= 0 else size[0] - 1
            # Gets the right neighbor; if at the end, loops to the beginning
            right = j + 1 if j + 1 < size[0] else 0

            # state_ is the two binary digits representing the spins of the left and right neighbors
            state_ = grid[[left, right]]

            # state is a vector of two binary digits representing action of two agents
            # turn this into a one-hot joint-action vector
            state = joint_state(state_)

            qvals = qfunc(s=state.float().detach(),
                          theta=params[j],
                          layers=[(4, hid_layer), (hid_layer, 2)])

            qmax = torch.argmax(qvals, dim=0).detach().item()

            action = int(qmax)

            # We take the action in our temporary copy of the grid, grid_, and only once all
            # agents have taken actions, do we copy them into the main grid
            grid_[j] = action

            reward = get_reward(state_.detach(), action)

            # The target value is the Q value vector with the Q value associated with
            # the action taken replaced with the reward observed.
            with torch.no_grad():
                target = qvals.clone()
                target[action] = reward

            # The loss is the sum of the squared difference between the Q value vector
            loss = torch.sum(torch.pow(qvals - target, 2))

            # We keep track of the losses for each agent
            losses[j].append(loss.detach().numpy())

            loss.backward()

            # Manual gradient descent
            with torch.no_grad():
                params[j] = params[j] - learning_rate * params[j].grad
            params[j].requires_grad = True

        # Copies the contents of the temporary grid_ into the main grid vector
        with torch.no_grad():
            grid.data = grid_.data

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(losses).T)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig('loss.png')

    # Show the final grid
    plt.figure(figsize=(10, 5))
    plt.imshow(np.expand_dims(grid, axis=0))
    # plt.show()
    plt.savefig('final_grid.png')
    