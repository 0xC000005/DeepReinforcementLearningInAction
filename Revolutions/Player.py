import names
from collections import deque
import Models


class Player:
    """
    Player class

    Attributes:
        player_name (str): Player's name
        player_team (int): Player's team
        player_current_privilege (int): Player's privilege in the current turn
        player_current_action (int): Player's chosen in the current turn
        player_current_reward (int): Player's reward in the current turn
        player_currently_blocking (int): The action the player is blocking the opponent from choosing
        player_current_action_space (list): Player's current action space
        player_currently_first_hand (bool): Whether the player is the first hand or not
        player_history (list): Player's history

    Methods:
        init_player_current_attr: Initialize player before each game, resetting the current action, reward, and action space
        update_player_history: Update player's history, including action, reward, and action space
        blocking_players_action: Block the player's action by removing the input from the action space
            Input: action (int)
            Output: None
        update_current_action_space: Update the player's current action space, keep or remove the block action option  base
        on the current player's privilege and the privilege of the opponent
            Input: opponent_privilege (int)
            Output: None
        choose_action_to_take: Choose an action from the player's current action space
            Input: None
            Output: None
        __str__: Print out the player's information
    """

    def __init__(self, player_team: int, replay_buffer_size=3):
        self.player_name = str(names.get_full_name())
        self.player_team = player_team
        """
        We want agent to be able to memorize what they have done at the each round of the game, so we need to keep track
        of the players privilege, action, associated reward, action space, and whatever they are first hand or not, 
        at each round of the game.
        """
        self.player_current_privilege = None
        self.player_current_action = None
        self.player_current_reward = None
        self.player_currently_blocking = None
        self.player_currently_first_hand = False
        self.player_current_action_space = []

        self.player_history = []
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.model = Models.DoubleDQN_dynamic()

    def reset_player(self):
        self.player_current_privilege = None
        self.player_current_action = None
        self.player_current_reward = None
        self.player_currently_blocking = None
        self.player_currently_first_hand = False
        self.player_current_action_space = []
        self.player_history = []

    def init_player_current_attr(self, player_current_privilege: int):
        self.player_current_action = None
        self.player_current_reward = None
        self.player_currently_blocking = None
        # # 1: cooperate, 2: defect, 3: block action, 4: revolution
        # self.player_current_action_space = [1, 2, 3, 4]

        # TODO: changed the action space to only three to accommodate the DQN model and reduce the potential action
        #  space, but does it makes sense? A human player might choose not to block, but right now the player cannot not
        #  blocking the opponent

        # 0: cooperate, 1: defect, 2: revolution
        self.player_current_action_space = [0, 1, 2]
        self.player_current_privilege = player_current_privilege

    def update_player_history(self):
        # append the players complete current stage into the player's history
        history = {
            "player_current_privilege": self.player_current_privilege,
            "player_current_action": self.player_current_action,
            "player_current_reward": self.player_current_reward,
            "player_currently_blocking": self.player_currently_blocking,
            "player_current_action_space": self.player_current_action_space
        }

        # append the history to the player's history
        self.player_history.append(history)

    def blocking_players_action(self, player_to_block: 'Player', action_to_block: int = None, user_input: bool = False):
        if user_input:
            # get the current action space of the player to block
            player_to_block_action_space = player_to_block.player_current_action_space
            # ask the human player to choose which action to block from the opponent's action space
            print("Player {}'s current action space: {}".format(player_to_block.player_name,
                                                                player_to_block_action_space))
            # print the action space hint: 0: cooperate, 1: defect, 2: revolution
            print("Hint: 0: cooperate, 1: defect, 2: revolution")
            action_to_block = int(input("Block an action from player {}'s action space: {}: ".format(
                player_to_block.player_name, player_to_block_action_space)))
            # set the current blocking action to the action to block
            self.player_currently_blocking = action_to_block

            # remove the action to block from the opponent's action space
            player_to_block_action_space.remove(action_to_block)

        else:
            player_to_block_action_space = player_to_block.player_current_action_space

            # set the current blocking action to the action to block
            self.player_currently_blocking = action_to_block

            # remove the action to block from the opponent's action space
            player_to_block_action_space.remove(action_to_block)

    # Removed to accommodate the DQN model since now blocking is not a optional action to take anymore
    # def update_current_action_space(self, opponent_privilege: int):
    #     if self.player_current_privilege < opponent_privilege:
    #         self.player_current_action_space.remove(3)

    def take_action(self, action: int = None, user_input: bool = False):
        # from the current action space, choose an action
        if user_input:
            print("Player {}'s current action space: {}".format(self.player_name, self.player_current_action_space))
            # print the action space hint: 0: cooperate, 1: defect, 2: revolution
            print("Hint: 0: cooperate, 1: defect, 2: revolution")
            action = int(input("Player {}'s action: ".format(self.player_name)))

            # set the user current action to the action taken
            self.player_current_action = action

        else:
            self.player_current_action = action

        # check if the current action taken is within the current action space, it not, raise a error
        if self.player_current_action not in self.player_current_action_space:
            raise ValueError("Player {}'s current action {} is not in the current action space {}".format(
                self.player_name, self.player_current_action, self.player_current_action_space))

    def __str__(self):
        # printout the player's information in a sublist format with \t indentation
        return ("\tPlayer Name: {}\n\t\tPlayer Team: {}\n\t\tPlayer Current Privilege: {}\n\t\tPlayer Current Action: "
                "{}"
                "\n\t\tPlayer Current Reward: {}\n\t\tPlayer Currently Blocking: {}\n\t\tPlayer Current Action Space: "
                "{}"
                "\n\t\tPlayer History: {}").format(self.player_name, self.player_team, self.player_current_privilege,
                                                   self.player_current_action, self.player_current_reward,
                                                   self.player_currently_blocking, self.player_current_action_space,
                                                   self.player_history)

    def __repr__(self):
        # display player name and the team they are on
        return "{}, Team: {}".format(self.player_name, self.player_team)
