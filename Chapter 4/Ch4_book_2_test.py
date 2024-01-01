import gymnasium as gym
import tianshou as ts
import torch
import numpy as np
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer

task = 'LunarLander-v2'
eps_test = 0.01
eps_train = 0.73
buffer_size = 100000
lr = 0.013
gamma = 0.99
n_step = 4
target_update_freq = 500
epoch = 10
step_per_epoch = 80000
step_per_collect = 16
update_per_step = 0.0625
batch_size = 128
hidden_sizes = [128, 128, 128]
dueling_q_hidden_sizes = [128, 128]
dueling_v_hidden_sizes = [128, 128]
training_num = 16
test_num = 100
render = 1 / 60

env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

# seed  (skip)

# model
Q_param = {"hidden_sizes": dueling_q_hidden_sizes}
V_param = {"hidden_sizes": dueling_v_hidden_sizes}
net = Net(state_shape=state_shape,
          action_shape=action_shape,
          hidden_sizes=[128, 128, 128],
          dueling_param=(Q_param, V_param))

optim = torch.optim.Adam(net.parameters(), lr=lr)

policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=gamma,
    action_space=env.action_space,
    estimation_step=n_step,
    target_update_freq=target_update_freq
)

train_collector = Collector(policy,
                            train_envs,
                            VectorReplayBuffer(buffer_size, training_num),
                            exploration_noise=True)
test_collector = Collector(policy,
                           test_envs,
                           exploration_noise=True)  # because DQN uses epsilon-greedy method


def save_best_fn(policy):
    torch.save(policy.state_dict(), 'best_dqn.pth')


def stop_fn(mean_rewards):
    return mean_rewards >= env.spec.reward_threshold


def train_fn(epoch, env_step):  # exp decay
    eps = max(eps_train * (1 - 5e-6) ** env_step, eps_test)
    policy.set_eps(eps)


def test_fn(epoch, env_step):
    policy.set_eps(eps_test)


# result = ts.trainer.OffpolicyTrainer(
#     policy=policy,
#     train_collector=train_collector,
#     test_collector=test_collector,
#     max_epoch=epoch,
#     step_per_epoch=step_per_epoch,
#     step_per_collect=step_per_collect,
#     episode_per_test=test_num,
#     batch_size=batch_size,
#     update_per_step=update_per_step,
#     train_fn=train_fn,
#     test_fn=test_fn,
#     stop_fn=stop_fn,
#     save_fn=save_best_fn,
# ).run()

# print(f'Finished training! Use {result["duration"]}')

# load the best_dqn.pth
policy.load_state_dict(torch.load('best_dqn.pth'))
policy.eval()
policy.set_eps(eps_test)
env = gym.make(task, render_mode='human')
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=10, render=1 / 60)
