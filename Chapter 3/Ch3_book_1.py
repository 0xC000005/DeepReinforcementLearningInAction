import numpy as np
import torch
from tqdm import tqdm
from Gridworld import Gridworld
import random
import matplotlib.pyplot as plt


def train(epsilon, model, mode, loss_function, optimizer, gamma, action_set):
    global losses
    epochs = 1000
    losses = []
    for i in tqdm(range(epochs)):
        # initialize gridworld, for each epoch, we start a new game
        game = Gridworld(size=4, mode=mode)
        # get the state representation of the current gridworld as a vector, plus normalized random noise
        init_state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        init_state = torch.from_numpy(init_state_).float()
        game_over = False

        current_state = init_state

        while not game_over:
            # get the expected reward for each action for the current state
            q_values_ = model(current_state)
            q_values = q_values_.data.numpy()[0]

            # epsilon greedy exploration
            if random.random() < epsilon:
                # choose a random action
                action_taken_ = np.random.randint(0, 4)
            else:
                # choose the action with the highest expected reward
                action_taken_ = np.argmax(q_values)

            action_taken = action_set[action_taken_]
            # apply the action to the gridworld, get the reward and the next state
            game.makeMove(action_taken)

            previous_state = current_state
            current_state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            current_state = torch.from_numpy(current_state_).float()

            current_reward = game.reward()

            with torch.no_grad():
                # get the expected reward for each action for the next state
                q_values_after_move_ = model(current_state.reshape(1, 64))

            # get the maximum expected reward for the next state
            max_q_values_after_move = torch.max(q_values_after_move_)

            if current_reward == -1:
                # if the game is not over, the target_q_value the q value of the previous state
                target_q_value = current_reward + gamma * max_q_values_after_move
            else:
                # if the game is over, the target_q_value is only the reward
                target_q_value = current_reward

                # set the game_over flag to True
                game_over = True

            # calculate the target_q_value the q value of the previous state
            target_q_value = torch.Tensor([target_q_value]).detach()
            # target_q_value the q value of the previous state
            predicted_q_value = q_values_.squeeze()[action_taken_]

            # calculate the loss
            loss = loss_function(target_q_value, predicted_q_value)
            optimizer.zero_grad()
            loss.backward()

            # record the loss for performance monitoring
            losses.append(loss.item())
            optimizer.step()

            # decrease the epsilon each epoch
            if epsilon > 0.1:
                epsilon -= 1 / epochs

    return model, losses


def test_model(model, mode='static', display=True):
    move_counter = 0
    test_game = Gridworld(mode=mode)
    init_state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    init_state = torch.from_numpy(init_state_).float()

    current_state = init_state
    game_over = 1

    if display:
        print("Initial State:")
        print(test_game.display())

    while game_over == 1:
        q_values_ = model(current_state)
        q_values = q_values_.data.numpy()
        chosen_action_ = np.argmax(q_values)
        chosen_action = action_set[chosen_action_]

        if display:
            print('Move #: %s; Taking action: %s' % (move_counter, chosen_action))

        test_game.makeMove(chosen_action)

        previous_state = current_state

        current_state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        current_state = torch.from_numpy(current_state_).float()

        if display:
            print(test_game.display())

        current_reward = test_game.reward()

        if current_reward != -1:
            if current_reward > 0:
                game_over = 2
                if display:
                    print("Game won! Reward: %s" % (current_reward,))
            else:
                game_over = 0
                if display:
                    print("Game LOST. Reward: %s" % (current_reward,))

        move_counter += 1
        if move_counter > 15:
            if display:
                print("Game lost; too many moves.")
            break
    win = True if game_over == 2 else False
    return win


if __name__ == '__main__':
    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )

    loss_function = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 0.3

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r'
    }

    model, losses = train(epsilon, model, 'random', loss_function, optimizer, gamma, action_set)

    # plot the losses
    plt.plot()
    # add a title
    plt.title('Loss per epoch')
    # add x and y labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plot the losses
    plt.plot(losses)
    plt.show()

    test_model(model, mode='random', display=True)
