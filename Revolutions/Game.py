import random
import Player

"""
Static functions for Game to use
"""


# TODO: add rewards for illegal actions
def calculate_payoff(first_hand: Player.Player, second_hand: Player.Player):
    # if any player choose to revolution, then both player receives -1
    if first_hand.player_current_action == 2 or second_hand.player_current_action == 2:
        return -1, -1

    # if both player choose to cooperate, then first hand receives 2 * its current privilege, and the second hand
    # receives 1
    if first_hand.player_current_action == 0 and second_hand.player_current_action == 0:
        # TODO: if the first hand player has a privilege of 0, then the power multiplier should be 1 to avoid 0 reward
        power_multiplier = max(first_hand.player_current_privilege, 1)
        return 2 * power_multiplier, 1

    # if both player choose to defect, then first hand receives 1, and the second hand receives 2
    if first_hand.player_current_action == 1 and second_hand.player_current_action == 1:
        return 1, 2

    # if one player choose to cooperate and the other choose to defect, both player receives 0
    if first_hand.player_current_action != second_hand.player_current_action:
        return 0, 0

    raise ValueError("Invalid action pair: {} and {}".format(first_hand.player_current_action,
                                                             second_hand.player_current_action))


class Game:
    """
    Game class

    Attributes:
        game_player_1 (Player): Player 1
        game_player_2 (Player): Player 2

    Methods:
        first_hand: Determine which player will go first
        init_players: Initialize the players
        update_player_current_reward: Update the player's current reward
        play: Play the game
        calculate_payoff: Calculate the payoff for both player
    """

    def __init__(self, player_1, player_2, debug=False):
        self.game_player_1 = player_1
        self.game_player_2 = player_2
        self.debug = debug

    # TODO: increase the chance of the second player getting the first hand to increase the chance of revolting
    def first_hand(self):
        """
        This function uses two uniform random variables to determine which player will go first. The range of the two
        random variables are [0, player1_privilege] and [0, player2_privilege], respectively. The player with the
        smaller random variable will go first. If the two random variables are equal, then the function will be
        called recursively.

        :return: player_1 or player_2
        """

        player_1_privilege = self.game_player_1.player_current_privilege
        player_2_privilege = self.game_player_2.player_current_privilege

        if player_1_privilege == player_2_privilege:
            # each player has a 50% chance of going first
            player_1_random_variable = random.uniform(0, 1)
            player_2_random_variable = random.uniform(0, 1)
        else:
            # generate two random variables from the uniform distribution, with the range of [0, player_privilege]
            player_1_random_variable = random.uniform(0, player_1_privilege)
            player_2_random_variable = random.uniform(0, player_2_privilege)

        # print out the two random variables\
        if self.debug:
            print("Player 1's random variable: {}".format(player_1_random_variable))
            print("Player 2's random variable: {}".format(player_2_random_variable))

        # if the two random variables are equal, then call the function recursively
        if player_1_random_variable == player_2_random_variable:
            return self.first_hand()

        # the one with larger random variable will go first
        if player_1_random_variable > player_2_random_variable:
            return self.game_player_1
        else:
            return self.game_player_2

    def init_players_for_game(self, player_1_current_privilege_from_team: int,
                              player_2_current_privilege_from_team: int):
        # initialize the player's all attributes associated with the current game
        self.game_player_1.init_player_current_attr(player_current_privilege=player_1_current_privilege_from_team)
        self.game_player_2.init_player_current_attr(player_current_privilege=player_2_current_privilege_from_team)

        # this block is removed since blocking is no longer an action but mandatory
        # # update the player's action space to remove the blocking action for the second-hand player
        # self.game_player_1.update_current_action_space(opponent_privilege=self.game_player_2.player_current_privilege)
        # self.game_player_2.update_current_action_space(opponent_privilege=self.game_player_1.player_current_privilege)

    def update_player_current_reward(self, first_hand: Player.Player, second_hand: Player.Player):
        player_1_reward, player_2_reward = calculate_payoff(first_hand=first_hand, second_hand=second_hand)
        first_hand.player_current_reward = player_1_reward
        second_hand.player_current_reward = player_2_reward

    def play(self,
             player_1_current_privilege_from_team: int,
             player_2_current_privilege_from_team: int,
             user_input: bool = False):

        # initialize the players
        self.init_players_for_game(player_1_current_privilege_from_team=player_1_current_privilege_from_team,
                                   player_2_current_privilege_from_team=player_2_current_privilege_from_team)

        first_hand = self.first_hand()
        second_hand = self.game_player_1 if first_hand == self.game_player_2 else self.game_player_2

        # first hand chooses its action
        first_hand.take_action(user_input=user_input)

        # # if the first hand chooses to block action, ask the first player to choose which action to block
        # if first_hand.player_current_action == 3:
        #     first_hand.blocking_players_action(action_to_block=None,
        #                                        player_to_block=second_hand, user_input=user_input)
        #
        #     # first-hand player chooses its action other than blocking again
        #     first_hand.take_action(user_input=user_input)

        # to accommodate the DQN model, right now the first hand player will always choose to block the second hand's
        # action
        first_hand.blocking_players_action(action_to_block=None, player_to_block=second_hand, user_input=user_input)

        # second hand chooses its action
        second_hand.take_action(user_input=user_input)

        # calculate the reward for both players
        self.update_player_current_reward(first_hand=first_hand, second_hand=second_hand)

        # update the player's history
        first_hand.update_player_history()
        second_hand.update_player_history()
