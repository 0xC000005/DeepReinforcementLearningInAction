import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the logging.csv file
def read_logging_csv(filename='logging.csv'):
    logging_data = pd.read_csv(filename)
    # drop the first row
    logging_data = logging_data.drop([0])
    return logging_data


# split the logging_data into n chunks
def split_logging_data(logging_data, n):
    return np.array_split(logging_data, n)


def get_average_action_and_blocking_statistics_over_current_chunk(logging_data) -> (list, list):
    # iterate through the logging_data and get the average action statistics over the current chunk
    total_action_statistics = [0] * 3
    total_blocking_statistics = [0] * 3
    for index, row in logging_data.iterrows():
        action_statistics = row["Action"]
        blocking_statistics = row['Blocking']

        # turn the string into a num list
        action_statistics = action_statistics[1:-1].split(',')
        action_statistics = [int(i) for i in action_statistics]

        blocking_statistics = blocking_statistics[1:-1].split(',')
        blocking_statistics = [int(i) for i in blocking_statistics]

        total_action_statistics[0] += action_statistics[0]
        total_action_statistics[1] += action_statistics[1]
        total_action_statistics[2] += action_statistics[2]

        total_blocking_statistics[0] += blocking_statistics[0]
        total_blocking_statistics[1] += blocking_statistics[1]
        total_blocking_statistics[2] += blocking_statistics[2]
    average_action_statistics = [total_action_statistics[0] / len(logging_data),
                                 total_action_statistics[1] / len(logging_data),
                                 total_action_statistics[2] / len(logging_data)]
    average_blocking_statistics = [total_blocking_statistics[0] / len(logging_data),
                                   total_blocking_statistics[1] / len(logging_data),
                                   total_blocking_statistics[2] / len(logging_data)]

    # # normalize the action statistics and the blocking statistics
    # total_action_statistics = sum(total_action_statistics)
    # total_blocking_statistics = sum(total_blocking_statistics)
    # average_action_statistics = [i / total_action_statistics for i in average_action_statistics]
    # average_blocking_statistics = [i / total_blocking_statistics for i in average_blocking_statistics]
    #
    # # convert to the exponential form to exaggerate the difference
    # average_action_statistics = [i ** 2 for i in average_action_statistics]
    # average_blocking_statistics = [i ** 2 for i in average_blocking_statistics]

    return average_action_statistics, average_blocking_statistics


def get_the_society_distribution_statistics(logging_data):
    society_distribution_statistics = [0] * 5
    for index, row in logging_data.iterrows():
        team_statistics = row["Teams"]
        team_statistics = team_statistics[1:-1].split(',')
        team_statistics = [abs(float(i)) for i in team_statistics]

        greater_than_100_counter = 0
        for i in team_statistics:
            if i >= 100:
                greater_than_100_counter += 1
        society_distribution_statistics[greater_than_100_counter] += 1
    return society_distribution_statistics


if __name__ == '__main__':
    logging_data_full = read_logging_csv('logging_1.csv')
    logging_data_split = split_logging_data(logging_data_full, 9999)

    # create a 2 rows 1 column figures
    fig, axs = plt.subplots(4, 1)

    statistics_table = None
    society_distribution_table = None

    # iterate through the logging_data_split and plot the action and blocking statistics
    for i in range(len(logging_data_split)):
        logging_data = logging_data_split[i]
        average_action_statistics, average_blocking_statistics = (
            get_average_action_and_blocking_statistics_over_current_chunk(logging_data))
        society_distribution_statistics = get_the_society_distribution_statistics(logging_data)

        if statistics_table is None:
            # create a dataframe with 6 columns
            statistics_table = pd.DataFrame(columns=['Action 1',
                                                     'Action 2',
                                                     'Action 3',
                                                     'Blocking 1',
                                                     'Blocking 2',
                                                     'Blocking 3'])

            society_distribution_table = pd.DataFrame(columns=['0',
                                                               '1',
                                                               '2',
                                                               '3',
                                                               '4'])
        else:
            statistics_table.loc[i] = average_action_statistics + average_blocking_statistics
            society_distribution_table.loc[i] = society_distribution_statistics

        statistics_table.loc[i] = average_action_statistics + average_blocking_statistics

    statistics_table = statistics_table.transpose()
    society_distribution_table = society_distribution_table.transpose()

    # plot the change of the first 3 rows over time
    # Plot the action statistics over time
    axs[0].plot(statistics_table.loc['Action 1'], label='Cooperate')
    axs[0].plot(statistics_table.loc['Action 2'], label='Defect')
    axs[0].plot(statistics_table.loc['Action 3'], label='Revolution')
    axs[0].set_title('Action Statistics Over Time')
    # set the legend position to be outside of the plot
    axs[0].legend(loc='right')
    # add a x-axis label
    axs[0].set_xlabel('Epoch')

    # plot the change of the last 3 rows over time
    # Plot the blocking statistics over time
    axs[1].plot(statistics_table.loc['Blocking 1'], label='Blocking Cooperate')
    axs[1].plot(statistics_table.loc['Blocking 2'], label='Blocking Defect')
    axs[1].plot(statistics_table.loc['Blocking 3'], label='Blocking Revolution')
    axs[1].set_title('Blocking Statistics Over Time')
    axs[1].legend(loc='right')
    axs[1].set_xlabel('Epoch')

    revolution_statistics = logging_data_full['Great Chaos']
    # convert the revolution statistics into a a int list
    revolution_statistics = [int(i) for i in revolution_statistics]
    # plot the moving average of the revolution of the logging_data_full over time
    axs[2].plot(pd.Series(revolution_statistics).rolling(100).mean())
    axs[2].set_title('Revolution Occurrence Over Time')
    axs[2].set_ylabel('Moving Average of 1000 Epochs')
    axs[2].set_xlabel('Epoch')

    # plot the society distribution statistics over time
    axs[3].plot(society_distribution_table.loc['0'], label='0')
    axs[3].plot(society_distribution_table.loc['1'], label='1')
    axs[3].plot(society_distribution_table.loc['2'], label='2')
    axs[3].plot(society_distribution_table.loc['3'], label='3')
    axs[3].plot(society_distribution_table.loc['4'], label='4')
    axs[3].set_title('How many teams are rich at the end of each epoch?')
    axs[3].legend(loc='right')
    axs[3].set_xlabel('Epoch')

    # increase the space between the two subplots
    fig.subplots_adjust(hspace=1)

    # increase the size of the figure
    fig.set_size_inches(18.5, 10.5)

    plt.show()

    # save the figure
    fig.savefig('action_blocking_statistics.png')
