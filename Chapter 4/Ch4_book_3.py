import gymnasium as gym
import tianshou as ts
import torch
import numpy as np
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import NoisyLinear

# detect device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


task = 'LunarLander-v2'
eps_test = 0.01
eps_train = 0.1
buffer_size = 100000
lr = 0.013
gamma = 0.99
num_atoms = 51
v_min = -10
v_max = 10
noisy_std = 0.1
n_step = 4
target_update_freq = 320
epoch = 100
step_per_epoch = 8000
step_per_collect = 8
update_per_step = 0.0625
batch_size = 128
hidden_sizes = [128, 128, 128]
# dueling_q_hidden_sizes = [128, 128]
# dueling_v_hidden_sizes = [128, 128]
training_num = 16
test_num = 100
render = 1 / 60
alpha = 0.6
beta = 0.4
beta_final = 1.0


env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

# seed  (skip)


def noisy_linear(x, y):
    return NoisyLinear(x, y, noisy_std)

# model
net = Net(state_shape=state_shape,
          action_shape=action_shape,
          hidden_sizes=[128, 128, 128],
          device=device,
          softmax=True,
          num_atoms=num_atoms,
          dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
          ).to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

policy = ts.policy.RainbowPolicy(
    model=net,
    optim=optim,
    discount_factor=gamma,
    action_space=env.action_space,
    num_atoms=num_atoms,
    v_min=v_min,
    v_max=v_max,
    estimation_step=n_step,
    target_update_freq=target_update_freq,
).to(device)

train_collector = Collector(policy,
                            train_envs,
                            VectorReplayBuffer(buffer_size, training_num),
                            exploration_noise=True)
test_collector = Collector(policy,
                           test_envs,
                           exploration_noise=True)  # because DQN uses epsilon-greedy method
train_collector.collect(n_step=batch_size * training_num)

def save_best_fn(policy):
    torch.save(policy.state_dict(), 'best_dqn.pth')


# def stop_fn(mean_rewards):
#     return mean_rewards >= env.spec.reward_threshold


def train_fn(epoch, env_step):  # exp decay
    if env_step <= 10000:
        policy.set_eps(eps_train)
    elif env_step <= 500000:
        eps = eps_train - (env_step - 100000) / 400000 * (0.9 * eps_train)
        policy.set_eps(eps)
    else:
        policy.set_eps(0.1 * eps_train)



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
#     update_per_step=1 / step_per_collect,
#     train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
#     test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
#     stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
# ).run()


result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=epoch,
    step_per_epoch=step_per_epoch,
    step_per_collect=step_per_collect,
    episode_per_test=test_num,
    batch_size=batch_size,
    update_per_step=update_per_step,
    train_fn=train_fn,
    test_fn=test_fn,
    # stop_fn=stop_fn,
    save_fn=save_best_fn,
).run()

print(f'Finished training! Use {result["duration"]}')

# policy.eval()
# policy.set_eps(eps_test)
# env = gym.make(task, render_mode='human')
# collector = ts.data.Collector(policy, env, exploration_noise=True)
# collector.collect(n_episode=10, render=1 / 60)

if __name__ == '__main__':
    pass

