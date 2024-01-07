import Revolution
import random
import math
import pandas as pd
import json
import datetime


def get_wealth_disparity(teams_score_statistics_table: list) -> float:
    # calculate the wealth disparity
    #  = (max - min) / max
    return max(teams_score_statistics_table) - min(teams_score_statistics_table)


class MultiAgentEnv:
    def __init__(self, num_agents=40, num_teams=4, advantage_bonus=1.5, failed_revolution_penalty=0):
        self.num_agents = num_agents
        self.num_teams = num_teams
        # randomly assign each team with an initial privilege from the range of [1, 10]
        self.init_privilege_per_team = [random.randint(1, 10) for _ in range(self.num_teams)]
        self.teams = Revolution.generate_teams(number_of_players_per_team=int(self.num_agents / self.num_teams),
                                               init_privilege_per_team=self.init_privilege_per_team)
        self.agents = Revolution.get_flatten_player_list(teams=self.teams)
        self.pairings = []
        self.revolution_count = 0
        self.agent_rewards = [0] * self.num_agents
        self.advantage_bonus = advantage_bonus
        self.failed_revolution_penalty = failed_revolution_penalty
        self.logging = None
        self.great_chaos_count = 0
        self.logging_filename = 'log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'

    def reset(self):
        """
        Reset the environment by re-initializing the teams
        Emptying pairings, revolution count, and agent rewards
        """
        self.pairings = []
        self.revolution_count = 0
        self.agent_rewards = [0] * self.num_agents
        for team_idx in range(len(self.teams)):
            team = self.teams[team_idx]
            init_privilege_of_current_team = random.randint(1, 10)
            team.reset_team(init_privilege=init_privilege_of_current_team)
            self.init_privilege_per_team[team_idx] = init_privilege_of_current_team

    def get_total_rewards_from_all_agents(self):
        """
        Get the total rewards from all agents
        """
        teams_current_total_reward_list = Revolution.get_teams_current_total_reward_list(teams=self.teams)
        return sum(teams_current_total_reward_list)

    def new_turn(self):
        """
        This function is called at the beginning of each turn. It updates the teams' current total rewards, historical
        total rewards, and privilege.
        """
        # get the player pairs
        self.pairings = Revolution.random_pair_players(teams=self.teams)

    # TODO: How is this redistribution calculated?
    def check_revolution(self, revolution_percentage=0.5, revolution_cost=1):
        great_chaos = False
        redistribution = None
        flatten_player_list = Revolution.get_flatten_player_list(teams=self.teams)
        self.revolution_count = Revolution.count_occurrence_of_revolution_within_current_round(
            players=flatten_player_list)
        if self.revolution_count > self.num_agents * revolution_percentage:
            total_rewards = self.get_total_rewards_from_all_agents()
            redistributed_reward_for_each_agent = math.floor(total_rewards / self.num_agents)
            redistribution = redistributed_reward_for_each_agent // self.num_agents
            great_chaos = True
        self.revolution_count = 0

        return great_chaos, redistribution

    def resolve_revolutions(self):
        great_chaos, redistribution = self.check_revolution()
        if great_chaos:
            self.great_chaos_count += 1
            for agent in self.agents:
                if len(agent.replay_buffer) > 0:
                    agent.replay_buffer[-1][2] = redistribution - agent.player_current_reward
                    agent.player_current_reward = redistribution
        else:
            for agent in self.agents:
                if len(agent.replay_buffer) > 0:
                    if agent.replay_buffer[-1][1] == 2:
                        # a failed revolution gets punished
                        agent.replay_buffer[-1][2] = self.failed_revolution_penalty

        return great_chaos

    def conclude_trial(self, round_num: int, advantaged_agents: list, disadvantaged_agents: list, done: int):
        for agent in advantaged_agents:
            if round_num > 1:
                agent.model.push_to_buffer(agent.replay_buffer[-2], restricted=False)
            if done:
                agent.model.push_to_buffer(agent.replay_buffer[-1], restricted=False)
        for agent in disadvantaged_agents:
            if round_num > 1:
                agent.model.push_to_buffer(agent.replay_buffer[-2], restricted=True)
            if done:
                agent.model.push_to_buffer(agent.replay_buffer[-1], restricted=True)

    def update_all_models(self, epoch):
        if epoch > 20:
            for agent in self.agents:
                loss1 = agent.model.train_td(restricted=True)
                loss2 = agent.model.train_td(restricted=False)

        if epoch % 200 == 0:
            for agent in self.agents:
                agent.model.update_target()

    def display_agent_score_board(self):
        # construct a table using dataframe with all players' attributes: player name, player team, player current
        # privilege, player current action, player current blocking action, player current reward
        player_name_list = []
        player_team_list = []
        player_current_privilege_list = []
        player_current_action_list = []
        player_current_blocking_action_list = []
        player_current_reward_list = []
        player_current_pairing_id_list = []
        for agent in self.agents:
            player_name_list.append(agent.player_name)
            player_team_list.append(agent.player_team)
            player_current_privilege_list.append(agent.player_current_privilege)
            player_current_action_list.append(agent.player_current_action)
            player_current_blocking_action_list.append(agent.player_currently_blocking)
            player_current_reward_list.append(agent.player_current_reward)
            # find the index of the current agent in the pairings list
            for pairing_id in range(len(self.pairings)):
                if self.pairings[pairing_id][0] == agent or self.pairings[pairing_id][1] == agent:
                    player_current_pairing_id_list.append(pairing_id)

        player_score_board = pd.DataFrame({
            "Name": player_name_list,
            "Team": player_team_list,
            "Privilege": player_current_privilege_list,
            "Action": player_current_action_list,
            "Blocking": player_current_blocking_action_list,
            "Reward": player_current_reward_list,
            "Pairing": player_current_pairing_id_list
        })

        # rank the player score board by the reward
        player_score_board = player_score_board.sort_values(by=['Reward'], ascending=False)

        # print clear line to empty the screen
        print(player_score_board.to_markdown())

    def display_team_score_board(self):
        # construct a table using dataframe with all teams current attributes: team name, team current total reward
        team_name_list = []
        team_current_total_reward_list = []
        team_current_total_privilege_list = []
        team_historical_total_rewards_list = []
        for team in self.teams:
            team_name_list.append(team.team_id)
            team_current_total_reward_list.append(team.team_current_total_reward)
            team_current_total_privilege_list.append(team.team_privilege)
            if len(team.team_historical_total_rewards) <= 3:
                team_historical_total_rewards_list.append(team.team_historical_total_rewards)
            else:
                team_historical_total_rewards_list.append(team.team_historical_total_rewards[-3:])

        team_score_board = pd.DataFrame({
            "Team": team_name_list,
            "Reward": team_current_total_reward_list,
            "Privilege": team_current_total_privilege_list,
            "History": team_historical_total_rewards_list
        })

        # rank the dataframe with reward
        team_score_board = team_score_board.sort_values(by=['Reward'], ascending=False)

        print(team_score_board.to_markdown())

    def get_action_blocking_and_illegal_action_statistics(self) -> (list, list, int):
        # get the flatten player list
        flatten_player_list = Revolution.get_flatten_player_list(teams=self.teams)

        action_statistics_table = [0] * 3
        blocking_statistics_table = [0] * 3
        illegal_action_count = 0

        for player in flatten_player_list:
            # iterate players history
            for history in player.player_history:
                # get the action and blocking action
                player_current_action = history['player_current_action']
                player_current_blocking_action = history['player_currently_blocking']
                illegal_action_count += history['player_current_illegal_action']

                # update the action statistics table
                action_statistics_table[player_current_action] += 1

                # update the blocking statistics table\
                if player_current_blocking_action is not None:
                    blocking_statistics_table[player_current_blocking_action] += 1

        return action_statistics_table, blocking_statistics_table, illegal_action_count

    def get_teams_score_statistics_table(self) -> list:
        # get the list of teams' current total reward
        return Revolution.get_teams_current_total_reward_list(teams=self.teams)

    def recording_log(self, current_epoch, current_epsilon, total_epoch, rounds_of_game_per_epoch):
        # construct a dataframe table with the following: epoch, epsilon, action_statistics_table,
        # blocking_statistics_table, great_chaos_count, teams_score_statistics_table and wealth_disparity
        action_statistics_table, blocking_statistics_table, illegal_action_count = self.get_action_blocking_and_illegal_action_statistics()
        teams_score_statistics_table = self.get_teams_score_statistics_table()

        logging_table = pd.DataFrame({
            "Action": [action_statistics_table],
            "Blocking": [blocking_statistics_table],
            "Illegal Action": illegal_action_count,
            "Great Chaos": [self.great_chaos_count],
            "Teams": [teams_score_statistics_table],
            "Wealth Disparity": [get_wealth_disparity(teams_score_statistics_table)],
            "Epsilon": [current_epsilon]
        })

        if self.logging is None:
            # create a metadata dictionary consist of the number of team, number of agents, the total epoch,
            # and for each epoch how many rounds of game is there

            metadata = {"Number of Teams": self.num_teams,
                        "Number of Agents": self.num_agents,
                        "Total Epoch": total_epoch,
                        "Rounds of Game per Epoch": rounds_of_game_per_epoch}

            # convert the metadata dictionary into a json
            metadata = json.dumps(metadata)

            # convert the metadata into a string
            metadata = str(metadata)

            metadata_df = pd.DataFrame({"Action": [metadata],
                                        "Blocking": 0,
                                        "Illegal Action": 0,
                                        "Great Chaos": 0,
                                        "Teams": 0,
                                        "Wealth Disparity": 0,
                                        "Epsilon": 0})

            self.logging = pd.concat([metadata_df, logging_table])

        else:
            self.logging = pd.concat([self.logging, logging_table], ignore_index=True)

        # # clear the screen
        # print(self.logging.to_markdown().split("\n")[-1])

        self.great_chaos_count = 0

        if current_epoch % 100 == 0:
            self.logging.to_csv(self.logging_filename)
