import os
os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用哪块GPU

import numpy as np
import tensorflow as tf
import scipy.signal
import matplotlib.pyplot as plt

from func_nn_ppo import func_nn_ppo
from HyperParameters import *

# ----

seed_generator = tf.random.set_seed(1337)

"""
## Functions and class
"""

def tf_get_mini_batches(obs_buf, act_buf,  adv_buf, ret_buf, logp_buf, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((
        obs_buf,
        act_buf,
        adv_buf,
        ret_buf,
        logp_buf,
    ))

    dataset = dataset.shuffle(buffer_size=len(obs_buf))  # 打乱数据
    dataset = dataset.batch(batch_size)  # 按批次分割数据
    return dataset

# def get_mini_batches(obs_buf, act_buf,  adv_buf, ret_buf, logp_buf, batch_size):
#     data_length = len(obs_buf)
#     indices = np.arange(data_length)
#     np.random.shuffle(indices) # 打乱 indices 的顺序

#     for start in range(0, data_length, batch_size): # 使用一个 for 循环，按 batch_size 的大小从打乱后的索引中逐步取出索引片段
#         end = start + batch_size
#         mini_batch_indices = indices[start:end]
#         yield (
#             obs_buf[mini_batch_indices],
#             act_buf[mini_batch_indices],
#             adv_buf[mini_batch_indices],
#             ret_buf[mini_batch_indices],
#             logp_buf[mini_batch_indices],
#         )


def discounted_cumulative_sums(x, discount): #  Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates (计算折扣累计和，用于计算奖励到期和优势估计)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer: # Buffer for storing trajectories (存储轨迹的缓冲区) 
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95): # Buffer initialization (缓冲区初始化)
        self.observation_buffer = np.zeros((size, observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_env_buffer = np.zeros(size, dtype=np.float32)
        self.return_env_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation_AI, action_AI_agent, reward_env, value, logprobability): # Append one step of agent-environment interaction (存储一次"代理-环境"交互的步骤)
        self.observation_buffer[self.pointer] = observation_AI
        self.action_buffer[self.pointer] = action_AI_agent
        self.reward_env_buffer[self.pointer] = reward_env
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0): # Finish the trajectory by computing advantage estimates and rewards-to-go (完成轨迹，通过计算优势估计和奖励到期)
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_env_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1] # 计算优势估计
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam) 

        self.return_env_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1] # 计算奖励到期

        self.trajectory_start_index = self.pointer

    def get(self): # Get all data of the buffer and normalize the advantages (获取所有缓冲区的数据)
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_env_buffer,
            self.logprobability_buffer,
        )

# def mlp(x, sizes, activation=tf.keras.activations.tanh, output_activation=None): # Build a feedforward neural network
#     # for size in sizes[:-1]:
#     #     x = tf.keras.layers.Dense(units=size, activation=activation)(x)
#     x = tf.keras.layers.Dense(64, activation="tanh")(x)
#     x = tf.keras.layers.Dense(64, activation="tanh")(x)

#     return tf.keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)
#     # return tf.keras.layers.Dense(64)(x)

def logprobabilities(action_logits_AI_agent, a): # Compute the log-probabilities of taking actions a by using the action_logits_AI_agent (i.e. the output of the actor) (计算动作 a 的对数概率)
    logprobabilities_all = tf.nn.log_softmax(action_logits_AI_agent)
    logprobability = tf.reduce_sum(tf.one_hot(a, num_actions) * logprobabilities_all, axis=1)
    return logprobability

# @tf.function是TensorFlow库中的一个装饰器，用于将普通的Python函数转换为TensorFlow图形函数(也称为计算图函数)。它的主要作用和目的是为了加速执行，
@tf.function # Sample action_AI_agent from actor
def sample_action(observation_AI):
    action_logits_AI_agent = actor(observation_AI)
    action_AI_agent = tf.squeeze(tf.random.categorical(action_logits_AI_agent, 1, seed=seed_generator), axis=1)
    return action_logits_AI_agent, action_AI_agent

@tf.function # Train the policy by maxizing the PPO-Clip objective
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer): # 训练策略网络
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(logprobabilities(actor(observation_buffer), action_buffer) - logprobability_buffer)
        min_advantage = tf.where(advantage_buffer > 0, (1 + clip_ratio) * advantage_buffer, (1 - clip_ratio) * advantage_buffer)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer))
    kl = tf.reduce_sum(kl)
    return kl

@tf.function # Train the value function by regression on mean-squared error
def train_value_function(observation_buffer, return_env_buffer): # 训练价值函数
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_env_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

"""
## Initializations  
"""




observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n
# print('\n')
# print('observation_dimensions:', observation_dimensions)
# print('num_actions:', num_actions)
# # print('\nenv.action_space:', env.action_space)
# print('\n')

obs_dict = env.reset() # 得到 reset 函数返回的字典值
# -- 把字典里面的各种物理量提取出来 --
both_agent_obs = obs_dict["both_agent_obs"] # 获取两个智能体的观察值
agent_obs_0 = both_agent_obs[0]  # 第一个智能体的观察值
agent_obs_1 = both_agent_obs[1]  # 第二个智能体的观察值
overcooked_state = obs_dict["overcooked_state"] # 获取当前的Overcooked状态
other_agent_env_idx = obs_dict["other_agent_env_idx"] # 获取另一个智能体的环境索引
# print('\n')
# print('=====> both_agent_obs is:', both_agent_obs)
# print('=====> agent_obs_0 is:', agent_obs_0)
# print('=====> agent_obs_1 is:', agent_obs_1)
# print('=====> overcooked_state is:', overcooked_state)
# print('=====> other_agent_env_idx is:', other_agent_env_idx)
# print('\n')

# ACTION_TO_CHAR = {
#     Direction.NORTH: "↑",
#     Direction.SOUTH: "↓",
#     Direction.EAST: "→",
#     Direction.WEST: "←",
#     STAY: "stay",
#     INTERACT: INTERACT,
# }





# ---- (CartPole) ----
# env = gym.make("CartPole-v1")
# observation_dimensions = env.observation_space.shape[0]
# num_actions = env.action_space.n
# # Initialize
# observation_AI, _ = env.reset()
# # print('=====> observation_AI init is:', observation_AI)
# episode_return_env, episode_length = 0, 0
# -------- end of env configuration --------

buffer = Buffer(observation_dimensions, steps_per_epoch) # Initialize the buffer (# 初始化缓冲区)
actor, critic = func_nn_ppo(observation_dimensions, num_actions)

# observation_input = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")
# action_logits_AI_agent = mlp(observation_input, list(mlp_hidden_sizes) + [num_actions])
# actor = tf.keras.Model(inputs=observation_input, outputs=action_logits_AI_agent, name='actor_keras')
# value = tf.squeeze(mlp(observation_input, list(mlp_hidden_sizes) + [1]), axis=1)
# critic = tf.keras.Model(inputs=observation_input, outputs=value, name='critic_keras')

# actor.summary()
# critic.summary()

# Initialize the policy and the value function optimizers
if bc_model_path_train == "./bc_runs_ireca/reproduce_train/asymmetric_advantages":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_aa)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_aa)



"""
## Train
"""

# -- 现在你可以使用这些变量进行后续的操作，比如决定哪个智能体行动，或者基于当前状态进行策略计算等
observation_AI = np.array(both_agent_obs[1 - other_agent_env_idx])
observation_HM = np.array(both_agent_obs[other_agent_env_idx])

episode_return_sparse, episode_return_shaped, episode_return_env = 0, 0, 0
episode_return_cau, episode_return_causal = 0, 0
episode_length = 0
count_step = 0

avg_return_shaped = []
avg_return_sparse = []
avg_return_env = []
avg_return_cau = []
avg_return_causal = []

for epoch in range(epochs):

    sum_return_sparse = 0
    sum_return_shaped = 0
    sum_return_env = 0
    sum_return_cau = 0
    sum_return_causal = 0

    sum_length = 0
    num_episodes = 0

    observation_AI = tf.reshape(observation_AI, (1, -1))
    observation_HM = tf.reshape(observation_HM, (1, -1))

    for t in range(steps_per_epoch):

        count_step += 1

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)
        action_logits_bc_agent = bc_model_train(observation_HM, training=False)        

        # ----------------------------------------------------------------------------------------------------------------------------
        ########## CALCULATE CAUSAL INFLUENCE ##########
        # ---- KL 散度公式的左边，知道 AI agent action 以后 --
        obs_both_agent_know_AI_action = env.dummy_step([action_AI_agent.numpy()[0], 4]) # 让 HM agent 不动
        observation_HM_know_AI_action = tf.reshape(obs_both_agent_know_AI_action[other_agent_env_idx], (1,-1)) # 得到只有 ai agent 行动以后的 HM 的 observation

        action_logits_HM_know_AI_action = bc_model_train(observation_HM_know_AI_action, training=False)
        action_probs_HM_know_AI_action = tf.nn.softmax(action_logits_HM_know_AI_action) # 这个其实就是边缘概率分布

        # ---- KL 散度公式的右边，不知道 AI agent action，其实就是用一开始的状态估计出来的 distribution  --
        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)

        # - 计算 CiMAR 的 KL divergence
        reward_cau = tf.keras.losses.KLDivergence()(action_probs_HM_know_AI_action, action_probs_bc_agent)

        # ----------------------------------------------------------------------------------------------------------------------------
        
#         action_HM_agent = tf.argmax(bc_model(observation_HM, training=True), axis=1) # 不用随机策略
        action_HM_agent = tf.argmax(action_logits_bc_agent, axis=1)

        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step([action_AI_agent.numpy()[0], action_HM_agent.numpy()[0]])

        coeff_reward_shaped = max(0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped*reward_shaped

        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1-other_agent_env_idx], (1,-1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1,-1))

        coeff_reward_shaped = max(0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped*reward_shaped

        # ----------------------------------------------------------------------------------------------------------------------------
        
        # reward_cau = tf.minimum(reward_cau, reward_env)

        reward_causal = reward_env + coeff_reward_cau*reward_cau

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped        
        episode_return_env += reward_env
        episode_return_cau += reward_cau
        episode_return_causal += reward_causal
        episode_length += 1

        # Get the value and log-probability of the action_AI_agent (获取动作的价值和对数概率)
        value_t = critic(observation_AI)
        logprobability_t = logprobabilities(action_logits_AI_agent, action_AI_agent)

        # Store obs, act, rew, v_t, logp_pi_t (存储观测、动作、奖励、价值和对数概率)
        buffer.store(observation_AI, action_AI_agent, reward_env, value_t, logprobability_t)

        # Update the observation_AI (更新观测)
        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

        # Finish trajectory if reached to a terminal state (如果达到终止状态则结束轨迹)
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            # print('count_step:', count_step)
            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return_shaped += episode_return_shaped
            sum_return_sparse += episode_return_sparse
            sum_return_env += episode_return_env
            sum_return_cau += episode_return_cau
            sum_return_causal += episode_return_causal
            sum_length += episode_length
            num_episodes += 1
            obs_dict = env.reset()
            observation_AI = tf.reshape(obs_dict["both_agent_obs"][1-other_agent_env_idx], (1,-1))
            observation_HM = tf.reshape(obs_dict["both_agent_obs"][other_agent_env_idx], (1,-1))
            episode_return_shaped, episode_return_sparse, episode_return_env = 0, 0, 0
            episode_return_cau, episode_return_causal = 0, 0
            episode_length = 0

    (obs_buf, act_buf, adv_buf, ret_buf, logp_buf) = buffer.get()

    for _ in range(iterations_train_policy):
        for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size):
        # for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size=batch_size):
            kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
            if kl > 1.5 * target_kl:
                break 
            train_value_function(obs_batch, ret_batch)

    print(f" [causal] Epoch: {epoch}. Mean Length: {sum_length / num_episodes}")
    print(f" Mean sparse: {sum_return_sparse / num_episodes}. Mean shaped: {sum_return_shaped / num_episodes}. Mean Env: {sum_return_env / num_episodes}. Mean cau: {sum_return_cau / num_episodes}. Mean causal: {sum_return_causal / num_episodes}.")
    # print('num_episodes:', num_episodes)

    avg_return_shaped.append(sum_return_shaped / num_episodes)
    avg_return_sparse.append(sum_return_sparse / num_episodes)
    avg_return_env.append(sum_return_env / num_episodes)
    avg_return_cau.append(sum_return_cau / num_episodes)
    avg_return_causal.append(sum_return_causal / num_episodes)

    if ((epoch+1) % 20 == 0) or ((epoch+1) == epochs) :
        np.save('./data_tmp/data_causal_return_stage.npy', avg_return_shaped)
        np.save('./data_tmp/data_causal_return_sparse.npy', avg_return_sparse)
        # np.save('./data_tmp/data_causal_return_env.npy',    avg_return_env)
        np.save('./data_tmp/data_causal_return_causal_reward.npy',    avg_return_cau)
        # np.save('./data_tmp/data_causal_return_causal.npy',  avg_return_causal)
        

f bc_model_path_train == "./bc_runs_ireca/reproduce_train/asymmetric_advantages":
    actor.save_weights("model_aa_actor_causal.h5")
    critic.save_weights("model_aa_critic_causal.h5")
        
# # ----Plot Figure----
# plt.figure()
# plt.plot(avg_return_env, markerfacecolor='none')
# # plt.xlabel('Index of data samples')
# plt.ylabel('avg_return_env')
# plt.legend(fontsize=12, loc='lower right')
# plt.savefig('./figs/causal_avg_return_env.pdf', format='pdf')  

# plt.show()