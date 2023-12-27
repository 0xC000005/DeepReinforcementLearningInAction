import random
import numpy as np
from matplotlib import pyplot as plt


# initialize a record list: a list of dictionary, first field is the
# number of times the current bandit is pulled, second field is the
# average reward of the current bandit
def init_record_of_all_bandits(number_of_bandits):
    record_of_all_actions = []
    for i in range(number_of_bandits):
        record_of_all_actions.append({'times_pulled': 0, 'average_reward': 0})
    return record_of_all_actions


RECORD_OF_ALL_ACTIONS = init_record_of_all_bandits(3)
PROBABILITIES_OF_ALL_BANDITS = [0.1, 0.5, 0.7]
EPSILON_EXPLORATION = 0.1


def update_record(bandit, reward):
    average_reward_until_now = RECORD_OF_ALL_ACTIONS[bandit]['average_reward']
    new_average_reward = (average_reward_until_now * RECORD_OF_ALL_ACTIONS[bandit]['times_pulled'] + reward) / (
            RECORD_OF_ALL_ACTIONS[bandit]['times_pulled'] + 1)
    RECORD_OF_ALL_ACTIONS[bandit]['times_pulled'] += 1
    RECORD_OF_ALL_ACTIONS[bandit]['average_reward'] = new_average_reward


def get_the_highest_expected_reward_bandit():
    highest_expected_reward_bandit = 0
    highest_expected_reward = 0
    for i in range(len(RECORD_OF_ALL_ACTIONS)):
        if RECORD_OF_ALL_ACTIONS[i]['average_reward'] > highest_expected_reward:
            highest_expected_reward_bandit = i
            highest_expected_reward = RECORD_OF_ALL_ACTIONS[i]['average_reward']
    return highest_expected_reward_bandit


def get_reward(probability, number_of_trial=10):
    temp_reward = 0
    for i in range(number_of_trial):
        if random.random() < probability:
            temp_reward += 1
    return temp_reward


def softmax(tau=2.12):
    total = sum([np.exp(RECORD_OF_ALL_ACTIONS[i]['average_reward'] / tau) for i in range(len(RECORD_OF_ALL_ACTIONS))])
    softmax_probability = [np.exp(RECORD_OF_ALL_ACTIONS[i]['average_reward'] / tau) / total for i in
                           range(len(RECORD_OF_ALL_ACTIONS))]
    return softmax_probability



if __name__ == '__main__':
    print('PROBABILITIES_OF_ALL_BANDITS: ', PROBABILITIES_OF_ALL_BANDITS)
    fig, ax = plt.subplots(2, 1)
    # increase the distance between the two subplots
    plt.subplots_adjust(hspace=0.5)

    # setup fig 1,1
    ax[0].set_xlabel('Plays')
    ax[0].set_ylabel('Avg Reward')
    ax[0].set_title('Average Reward vs Plays')
    running_average_reward_over_time = [0]

    for i in range(500):
        if random.random() > EPSILON_EXPLORATION:
            chosen_action = get_the_highest_expected_reward_bandit()
        else:
            chosen_action = np.random.choice(3)

        current_reward = get_reward(probability=PROBABILITIES_OF_ALL_BANDITS[chosen_action])
        update_record(chosen_action, current_reward)
        running_average_until_now = running_average_reward_over_time[-1]
        running_average_reward_over_time.append((running_average_until_now * i + current_reward) / (i + 1))

    ax[0].plot(running_average_reward_over_time)

    # re-initialize the record list
    RECORD_OF_ALL_ACTIONS = init_record_of_all_bandits(3)

    # setup fig 2,1
    ax[1].set_xlabel('Plays')
    ax[1].set_ylabel('Avg Reward')
    ax[1].set_title('Average Reward vs Plays')
    running_average_reward_over_time = [0]


    for i in range(500):
        probability_of_choosing_each_bandit = softmax()
        chosen_action = np.random.choice(3, p=probability_of_choosing_each_bandit)

        current_reward = get_reward(probability=PROBABILITIES_OF_ALL_BANDITS[chosen_action])
        update_record(chosen_action, current_reward)
        running_average_until_now = running_average_reward_over_time[-1]
        running_average_reward_over_time.append((running_average_until_now * i + current_reward) / (i + 1))

    ax[1].plot(running_average_reward_over_time)
    plt.show()
