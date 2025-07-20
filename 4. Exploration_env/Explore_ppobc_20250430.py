import numpy as np
import tensorflow as tf
import scipy.signal
import matplotlib.pyplot as plt

from func_nn_ppo import func_nn_ppo
from HyperParameters import *

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

policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)

seed_generator = tf.random.set_seed(0)


buffer = Buffer(observation_dimensions, steps_per_epoch) # Initialize the buffer (# 初始化缓冲区)
actor, critic = func_nn_ppo(observation_dimensions, num_actions)


"""
## Train
"""


avg_return_env = []
avg_return_sparse = []
accumulated_sparse = []

for epoch in range(epochs):

    sum_return_env = 0
    sum_return_sparse = 0
    sum_length = 0
    num_episodes = 0

    # observation_AI = tf.reshape(observation_AI, (1, -1))

    for t in range(steps_per_epoch):

        observation_AI = tf.reshape(observation_AI, (1, -1))

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)

        action_AI_agent_np = action_AI_agent.numpy().tolist()[0]
        if t%10 == 0:
            action_HM_agent_np = np.random.randint(0, 3)   #action_HM_agent.numpy().tolist()[0]
        else:
            action_HM_agent_np = 4
        # action_HM_agent_np = 4

        # ----
        action_np = [action_AI_agent_np, action_HM_agent_np]
        
        obs_dict_new, reward_env, reward_sparse, done, _ = env.step(action_np)

        observation_AI_new = obs_dict_new

        episode_return_env    += reward_env
        episode_return_sparse += reward_sparse
        episode_length += 1

        # Get the value and log-probability of the action_AI_agent (获取动作的价值和对数概率)
        value_t = critic(observation_AI)
        logprobability_t = logprobabilities(action_logits_AI_agent, action_AI_agent)

        # Store obs, act, rew, v_t, logp_pi_t (存储观测、动作、奖励、价值和对数概率)
        buffer.store(observation_AI, action_AI_agent, reward_env, value_t, logprobability_t)

        # Update the observation_AI (更新观测)
        observation_AI = observation_AI_new

        if done or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return_env    += episode_return_env
            sum_return_sparse += episode_return_sparse
            sum_length += episode_length
            num_episodes += 1
            obs_dict = env.reset()
            # print("env reset")
            observation_AI = tf.reshape(obs_dict, (1,-1))
            episode_return_env, episode_return_sparse, episode_length = 0, 0, 0

    (obs_buf, act_buf, adv_buf, ret_buf, logp_buf) = buffer.get()

    for _ in range(iterations_train_policy):
        # for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size=batch_size):
        for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size):
            kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
            if kl > 1.5 * target_kl:
                break 
            train_value_function(obs_batch, ret_batch)


    print(f" [ppobc] Epoch: {epoch}. num_episodes: {num_episodes}. Mean Length: ({sum_length / num_episodes})")
    print(f" Sum sparse: {sum_return_sparse}. ")

    avg_return_sparse.append(sum_return_sparse / num_episodes)
    accumulated_sparse.append(sum_return_sparse)

    np.save('./data/toy_ppobc_sparse.npy', accumulated_sparse)
    # env.render()


# # ----Plot Figure----
plt.figure()
plt.plot(accumulated_sparse, markerfacecolor='none')
# plt.xlabel('Index of data samples')
plt.ylabel('avg return')
plt.legend(fontsize=12, loc='lower right')
# plt.savefig('./figs/ppobc_avg_return_env.pdf', format='pdf')  


# plt.show()