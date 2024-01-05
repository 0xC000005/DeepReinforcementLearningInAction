import Team
import Player
import Game
import random
import math


def get_flatten_player_list(teams: list):
    """
    Flatten the list of players from all teams

    :param teams: a list of teams
    :return: a list of players from all teams
    """
    return [player for team in teams for player in team.team_players]


def random_pair_players(teams: list):
    """
    Return a list of randomly paired players from different teams ensuring all players are paired once.
    The idea is to flatten the list of players from all teams, randomly select two players from the flattened list,
    and remove the two players from the flattened list. Repeat the process until there are only two players left in
    the flattened list. If the last two players are from the same team, then recursively call the function. Otherwise,
    the last two players are considered to be a valid pair.

    :param teams: a list of teams, each containing a 'team_players' list
    :return: a list of player pairs
    """

    # TODO: This function is not very efficient, since it has to recursively generate random pairing schemas until a
    #  valid a valid one is found. A more efficient way is to generate a decision tree of all possible pairing
    #  schemas, and randomly select one from the decision tree.

    flatten_player_list = get_flatten_player_list(teams=teams)
    player_pairs = []

    while len(flatten_player_list) > 2:
        # randomly select two players from the flatten_player_list list
        temp_player_1 = random.choice(flatten_player_list)
        flatten_player_list.remove(temp_player_1)
        temp_player_2 = random.choice(flatten_player_list)
        while temp_player_1.player_team == temp_player_2.player_team:
            temp_player_2 = random.choice(flatten_player_list)
        flatten_player_list.remove(temp_player_2)
        player_pairs.append((temp_player_1, temp_player_2))

    # check if the last two players are from the same team
    if flatten_player_list[0].player_team == flatten_player_list[1].player_team:
        # recursively call the function
        return random_pair_players(teams=teams)

    player_pairs.append((flatten_player_list[0], flatten_player_list[1]))
    return player_pairs


def get_current_privilege_from_team(player: Player.Player, teams: list):
    """
    Get the current privilege of the player from the team

    :param player: the player
    :param teams: a list of teams
    :return: the current privilege of the player
    """
    for team in teams:
        if player in team.team_players:
            return team.team_privilege


def generate_teams(number_of_players_per_team: int, init_privilege_per_team: list):
    number_of_teams = len(init_privilege_per_team)
    # check if the number of players * the number of teams is divisible by 2
    if number_of_players_per_team * number_of_teams % 2 != 0:
        raise ValueError("The number of players * the number of teams must be divisible by 2")

    # iterate through the teams, creating players, with the team ID equal to the index of the team privilege in the
    # initial privilege list
    teams = []
    for i in range(number_of_teams):
        players = []
        for j in range(number_of_players_per_team):
            players.append(Player.Player(player_team=i))
        teams.append(Team.Team(team_id=i, team_init_privilege=init_privilege_per_team[i], players=players))

    return teams


def get_teams_current_total_reward_list(teams: list):
    """
    Get the points of all teams

    :param teams: a list of teams
    :return: a list of points of all teams
    """
    return [team.team_current_total_reward for team in teams]


def get_relative_teams_reward(player: 'Player', teams: list):
    # get the current total reward of all teams
    teams_current_total_reward_list = get_teams_current_total_reward_list(teams=teams)

    # if teams current total reward list is a zero list, check the privilege of all teams instead
    team_reward_all_zero = True
    for reward in teams_current_total_reward_list:
        if reward != 0:
            team_reward_all_zero = False
            break

    if team_reward_all_zero:
        teams_current_total_reward_list = [team.team_privilege for team in teams]

    # normalize the teams current total reward list
    min_val = min(teams_current_total_reward_list)
    max_val = max(teams_current_total_reward_list)
    normalized_team_rewards = []
    if max_val == min_val:
        normalized_team_rewards = [1 if x == max_val else 0 for x in teams_current_total_reward_list]
    else:
        normalized_team_rewards = [
            (x - min_val) / (max_val - min_val) for x in teams_current_total_reward_list
        ]

    # Compute the log difference
    if max_val == min_val:
        log_diff = 0  # or some other predefined value
    else:
        log_diff = math.log(max_val - min_val)

    # Reorder the list
    relative_team_reward = normalized_team_rewards.pop(player.player_team)
    normalized_team_rewards.sort(reverse=True)
    reordered_scores = (
            [relative_team_reward] + normalized_team_rewards + [log_diff / 10]
    )  # think about diff norm

    return reordered_scores


def count_occurrence_of_revolution_within_current_round(players: list):
    """
    Count the occurrence of revolution within the current round

    :param Teams: a list of teams
    :return: the number of revolutions within the current round
    """

    revolution_counter = 0
    for player in players:
        if player.player_current_action == 2:
            revolution_counter += 1

    return revolution_counter


if __name__ == '__main__':
    # initialize the teams
    teams = generate_teams(number_of_players_per_team=3, init_privilege_per_team=[10, 1])

    # get the player pairs
    player_pairs = random_pair_players(teams=teams)

    for pair in player_pairs:
        player_1 = pair[0]
        player_2 = pair[1]

        # create a game
        game = Game.Game(player_1=player_1, player_2=player_2)

        # re-initialize the players before each game with the current privilege from the team
        player_1_current_privilege_from_team = get_current_privilege_from_team(player=player_1, teams=teams)
        player_2_current_privilege_from_team = get_current_privilege_from_team(player=player_2, teams=teams)

        game.play(player_1_current_privilege_from_team=player_1_current_privilege_from_team,
                  player_2_current_privilege_from_team=player_2_current_privilege_from_team,
                  user_input=True)

    # iterate through all teams and update their histories after one round of the game
    for team in teams:
        team.update_team()
