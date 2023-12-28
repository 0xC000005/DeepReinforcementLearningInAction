import numpy as np
import torch
from tqdm import tqdm
from Gridworld import Gridworld
from collections import deque
import random
import matplotlib.pyplot as plt

# detect if there is a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print out the device name
print(device)

def train_with_experience_replay(epsilon, model, mode, loss_function, optimizer, gamma, action_set):
    model.to(device)
    epochs = 5000
    losses = []
    memory_size = 1000
    batch_size = 200
    replay = deque(maxlen=memory_size)
    max_moves = 50
    for i in tqdm(range(epochs)):
        game = Gridworld(size=4, mode=mode)
        init_state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        init_state = torch.from_numpy(init_state_).float().to(device)
        game_over = False
        current_state = init_state
        move_counter = 0
        while not game_over:
            move_counter += 1
            q_values_ = model(current_state)
            q_values = q_values_.data.cpu().numpy()
            if random.random() < epsilon:
                action_taken_ = np.random.randint(0, 4)
            else:
                action_taken_ = np.argmax(q_values)
            action_taken = action_set[action_taken_]
            game.makeMove(action_taken)
            previous_state = current_state
            current_state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            current_state = torch.from_numpy(current_state_).float().to(device)
            current_reward = game.reward()
            game_over = True if current_reward > 0 else False

            experience = (previous_state, action_taken_, current_reward, current_state, game_over)
            replay.append(experience)

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                previous_state_batch = torch.cat([s[0] for s in minibatch]).to(device)
                action_batch = torch.Tensor([s[1] for s in minibatch]).to(device)
                reward_batch = torch.Tensor([s[2] for s in minibatch]).to(device)
                current_state_batch = torch.cat([s[3] for s in minibatch]).to(device)
                game_over_batch = torch.Tensor([s[4] for s in minibatch]).to(device)

                recomputed_q_values_per_previous_stages_batch = model(previous_state_batch)
                with torch.no_grad():
                    recomputed_q_values_per_current_stages_batch = model(current_state_batch)

                target_q_values_for_learning = (reward_batch +
                                                gamma *
                                                torch.max(recomputed_q_values_per_current_stages_batch, dim=1)[0]
                                                * (1 - game_over_batch))

                old_q_values_for_learning = recomputed_q_values_per_previous_stages_batch.gather(
                    dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                loss = loss_function(old_q_values_for_learning, target_q_values_for_learning.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if current_reward != -1 or move_counter > max_moves:
                    game_over = True
                    move_counter = 0

    return losses, model

def test_model(model, mode='static', display=True):
    model.to(device)
    move_counter = 0
    test_game = Gridworld(mode=mode)
    init_state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    init_state = torch.from_numpy(init_state_).float().to(device)

    current_state = init_state
    game_over = 1

    if display:
        print("Initial State:")
        print(test_game.display())

    while game_over == 1:
        q_values_ = model(current_state)
        q_values = q_values_.data.cpu().numpy()
        chosen_action_ = np.argmax(q_values)
        chosen_action = action_set[chosen_action_]

        if display:
            print('Move #: %s; Taking action: %s' % (move_counter, chosen_action))

        test_game.makeMove(chosen_action)

        previous_state = current_state

        current_state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        current_state = torch.from_numpy(current_state_).float().to(device)

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
    model.to(device)
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

    losses, model = train_with_experience_replay(epsilon, model, 'random',
                                                 loss_function, optimizer, gamma, action_set)

    # plot the losses
    plt.figure()
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig('losses.png')

    test_model(model, mode='random', display=True)

    test_performance_with_experience_replay(model)
