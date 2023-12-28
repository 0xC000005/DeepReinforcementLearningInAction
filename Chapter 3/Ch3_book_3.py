import numpy as np
import torch
import copy
from tqdm import tqdm
from Gridworld import Gridworld
from collections import deque
import random
import matplotlib.pyplot as plt


def train_with_experience_replay_and_target_network(epsilon, model, target_model, mode, loss_function, optimizer, gamma,
                                                    action_set):
    epochs = 5000
    losses = []
    memory_size = 1000
    batch_size = 200
    replay = deque(maxlen=memory_size)
    max_moves = 50
    target_Q_sync_counter = 0
    for i in tqdm(range(epochs)):

        game = Gridworld(size=4, mode=mode)
        init_state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        init_state = torch.from_numpy(init_state_).float()
        if_game_over = False
        state_before_move = init_state
        move_counter = 0

        while not if_game_over:
            target_Q_sync_counter += 1
            move_counter += 1
            q_values_ = model(state_before_move)
            q_values = q_values_.data.numpy()
            if random.random() < epsilon:
                action_taken_ = np.random.randint(0, 4)
            else:
                action_taken_ = np.argmax(q_values)
            action_taken = action_set[action_taken_]
            game.makeMove(action_taken)

            state_after_move_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state_after_move = torch.from_numpy(state_after_move_).float()

            reward = game.reward()

            if_game_over = True if reward > 0 else False

            experience = (state_before_move, action_taken_, reward, state_after_move, if_game_over)

            replay.append(experience)

            state_before_move = state_after_move
            state_after_move = None

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)

                state_before_move_batch = torch.cat([s[0] for s in minibatch])
                action_batch = torch.Tensor([s[1] for s in minibatch])
                reward_batch = torch.Tensor([s[2] for s in minibatch])
                state_after_move_batch = torch.cat([s[3] for s in minibatch])
                if_game_over_batch = torch.Tensor([s[4] for s in minibatch])

                Q1 = model(state_before_move_batch)
                with torch.no_grad():
                    Q2 = target_model(state_after_move_batch)

                # Double DQN
                Y = reward_batch + gamma * ((1 - if_game_over_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                loss = loss_function(X, Y.detach())

                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if target_Q_sync_counter % sync_freq == 0:
                    target_model.load_state_dict(model.state_dict())

            if reward != -1 or move_counter > max_moves:
                if_game_over = True
                move_counter = 0

    return losses, model, target_model


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


def test_performance_with_experience_replay(model):
    max_games = 1000
    wins = 0
    for i in range(max_games):
        win = test_model(model, mode='random', display=False)
        if win:
            wins += 1
    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games, wins))
    print("Win percentage: {}".format(win_perc))


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

    # create a second model by making a deep copy of the original Q-network model
    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())
    sync_freq = 50

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

    losses, model, target_model = train_with_experience_replay_and_target_network(epsilon=epsilon,
                                                                                  model=model,
                                                                                  target_model=target_model,
                                                                                  mode='random',
                                                                                  loss_function=loss_function,
                                                                                  optimizer=optimizer,
                                                                                  gamma=gamma,
                                                                                  action_set=action_set)

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

    test_performance_with_experience_replay(model)
