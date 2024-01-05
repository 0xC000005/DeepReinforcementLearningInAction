import torch
import numpy as np
import Revolution
from MultiAgentEnv import MultiAgentEnv
from Models import DoubleDQN_dynamic
from tqdm import tqdm
from Game import Game

env = MultiAgentEnv()
net = DoubleDQN_dynamic()
epsilon = 0.7
epochs = 100000
revolution_percentage = 0.15
total_rewards = 0
wealth_disparity = 0


def action_space_to_one_hot(action_space):
    one_hot = [0] * 3
    for action in action_space:
        one_hot[action] = 1
    return one_hot


# TODO: starting with a large ep
for epoch in tqdm(range(epochs)):
# for epoch in range(epochs):
    env.reset()
    if epoch <= 5000:
        epsilon = 0.5
    if epoch > 7500:
        epsilon = epsilon * 0.99995
    done = 0

    # run the game for 100 rounds
    for i in range(100):

        if round == 99:
            done = 1

        # if epoch % 1000 == 0:
        #     print("Round {} of epoch {}".format(i, epoch))
        env.new_turn()
        advantage_agents = []
        disadvantage_agents = []

        for pair in env.pairings:
            agent_1, agent_2 = pair
            agent_1_team = agent_1.player_team
            agent_2_team = agent_2.player_team
            agent_1_current_privilege_from_team = Revolution.get_current_privilege_from_team(player=agent_1,
                                                                                             teams=env.teams)
            agent_2_current_privilege_from_team = Revolution.get_current_privilege_from_team(player=agent_2,
                                                                                             teams=env.teams)

            game = Game(player_1=agent_1, player_2=agent_2)
            game.init_players_for_game(player_1_current_privilege_from_team=agent_1_current_privilege_from_team,
                                       player_2_current_privilege_from_team=agent_2_current_privilege_from_team)

            advantaged_agent = game.first_hand()
            disadvantaged_agent = agent_1 if advantaged_agent == agent_2 else agent_2

            advantage_agents.append(advantaged_agent)
            disadvantage_agents.append(disadvantaged_agent)

            # TODO: wills action space only has 3 items, does that mean the advantaged agent will always choose to
            #  block?
            one_hot_advantaged_action_space = action_space_to_one_hot(advantaged_agent.player_current_action_space)

            # TODO: the use of the relative team rewards here is unclear, why 4 teams will results in 5 items in the
            #  array
            relative_team_rewards = Revolution.get_relative_teams_reward(player=advantaged_agent, teams=env.teams)

            # agent state = combined list of one hot action space and relative team reward
            advantaged_agent_state = one_hot_advantaged_action_space + relative_team_rewards

            advantaged_agent_state = torch.tensor(advantaged_agent_state).float().reshape(8, )
            advantaged_agent_action, opponent_action_to_block = advantaged_agent.model.take_action(
                state=advantaged_agent_state,
                restricted=False,
                epsilon=epsilon,
                action_mask=one_hot_advantaged_action_space)

            advantaged_illegal_current_move = advantaged_agent.take_action(action=advantaged_agent_action)

            if len(advantaged_agent.replay_buffer) != 0:
                # appending to the deque replay buffer
                advantaged_agent.replay_buffer[-1][3] = advantaged_agent_state

            advantaged_agent.blocking_players_action(player_to_block=disadvantaged_agent,
                                                     action_to_block=opponent_action_to_block)

            one_hot_disadvantaged_action_space = action_space_to_one_hot(
                disadvantaged_agent.player_current_action_space)

            disadvantaged_agent_state = (one_hot_disadvantaged_action_space +
                                         Revolution.get_relative_teams_reward(player=disadvantaged_agent,
                                                                              teams=env.teams))

            disadvantaged_agent_state = (torch.tensor(disadvantaged_agent_state).float().reshape(8, ))

            disadvantaged_agents_action = disadvantaged_agent.model.take_action(
                state=disadvantaged_agent_state,
                restricted=True,
                epsilon=epsilon,
                action_mask=one_hot_disadvantaged_action_space)

            disadvantaged_agent_illegal_move = disadvantaged_agent.take_action(action=disadvantaged_agents_action)

            if len(disadvantaged_agent.replay_buffer) != 0:
                # appending to the deque replay buffer
                disadvantaged_agent.replay_buffer[-1][3] = disadvantaged_agent_state

            game.update_player_current_reward(first_hand=advantaged_agent,
                                              second_hand=disadvantaged_agent,
                                              advantaged_agent_illegal_move=advantaged_illegal_current_move,
                                              disadvantaged_agent_illegal_move=disadvantaged_agent_illegal_move)

            advantaged_agent_reward = advantaged_agent.player_current_reward
            disadvantaged_agent_reward = disadvantaged_agent.player_current_reward

            # amplify the advantaged players reward received, good or bad
            advantaged_agent.player_current_reward *= env.advantage_bonus

            # TODO: unlike the old code, here we make the disadvantaged agent unable to block, rather than assigning
            #  it with a hugely negative reward

            advantaged_agent.replay_buffer.append([advantaged_agent_state,
                                                   advantaged_agent.player_current_action,
                                                   advantaged_agent.player_current_reward,
                                                   advantaged_agent_state,
                                                   done])

            disadvantaged_agent.replay_buffer.append([disadvantaged_agent_state,
                                                      disadvantaged_agent.player_current_action,
                                                      disadvantaged_agent.player_current_reward,
                                                      disadvantaged_agent_state,
                                                      done])

            advantaged_agent.update_player_history()
            disadvantaged_agent.update_player_history()

        # if epoch % 1000 == 0:
        #     print("")
        #     env.display_agent_score_board()

        # TODO: the logic is unclear, needs to have Will explaining the logic of revolution & redistribution
        revolution = env.resolve_revolutions()
        # if epoch % 1000 == 0:
        #     if revolution:
        #         print("-------------------REVOLUTION-------------------")
        #         env.display_agent_score_board()

        # iterate through all teams and update their histories after one round of the game
        for team in env.teams:
            team.update_team()

        # if epoch % 1000 == 0:
        #     env.display_team_score_board()

        env.conclude_trial(round_num=i,
                           advantaged_agents=advantage_agents,
                           disadvantaged_agents=disadvantage_agents,
                           done=done)

    env.display_logging_info(epoch=epoch, epsilon=epsilon)
    env.update_all_models(epoch=epoch)

if __name__ == '__main__':
    pass
