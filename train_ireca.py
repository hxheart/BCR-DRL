import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用哪块GPU

import numpy as np
import tensorflow as tf
import scipy.signal
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


from func_nn_ppo import func_nn_ppo
from HyperParameters import *


# ----

seed_generator = tf.random.set_seed(1337)

"""
## Functions and class
"""

# def tf_sorted_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, priority_vec, batch_size):
#     all_data = tf.stack([obs_buf, act_buf, adv_buf, ret_buf, logp_buf, priority_vec], axis=-1) # 将所有数据合并成一个大张量，最后一维对应各个特征    
#     indices = tf.argsort(priority_vec, direction='DESCENDING')  # 根据优先级向量排序索引 # 降序排序以获得最高优先级在前的索引
#     sorted_data = tf.gather(all_data, indices, axis=0) # 使用排序索引对数据重新排序
#     obs_sorted, act_sorted, adv_sorted, ret_sorted, logp_sorted, priority_sorted = tf.unstack(sorted_data, axis=-1) # 将排序后的数据分割为观察、动作等单独的张量
#     dataset = tf.data.Dataset.from_tensor_slices((obs_sorted, act_sorted, adv_sorted, ret_sorted, logp_sorted)) # 创建排序后数据的数据集
#     # dataset = dataset.shuffle(buffer_size=len(obs_sorted)) # 打乱数据（如果需要的话，但根据优先级排序后通常不再打乱）
#     dataset = dataset.batch(batch_size) # 分割数据集为批次    
#     return dataset


def tf_get_mini_batches(obs_buf, act_buf,  adv_buf, ret_buf, logp_buf, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((  # 使用TensorFlow的tf.data.Dataset.from_tensor_slices方法将输入的五个数据缓冲区转换为一个tf.data.Dataset对象。这个方法会将每个输入缓冲区的对应元素组合在一起，形成一个元组。
        obs_buf,
        act_buf,
        adv_buf,
        ret_buf,
        logp_buf,
    ))

    dataset = dataset.shuffle(buffer_size=len(obs_buf))  # 调用shuffle方法对数据集进行打乱。buffer_size参数指定了打乱的缓冲区大小，这里设置为数据集的长度。这意味着在打乱过程中，数据集的所有元素都会被打乱。
    dataset = dataset.batch(batch_size)  # 调用batch方法将数据集分割成批次。batch_size参数指定每个批次的大小。每次迭代时，数据集将返回一个包含batch_size个样本的批次
    return dataset #返回创建好的数据集对象，供后续训练过程中使用。

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
    def __init__(self, observation_dimensions, size, gamma, GAE_lam): # Buffer initialization (缓冲区初始化)
        self.obs_buf = np.zeros((size, observation_dimensions), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.value_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.GAE_lam = gamma, GAE_lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation_AI, action_AI_agent, reward_env, value, logprobability): # Append one step of agent-environment interaction (存储一次"代理-环境"交互的步骤)
        self.obs_buf[self.pointer] = observation_AI
        self.act_buf[self.pointer] = action_AI_agent
        self.reward_buf[self.pointer] = reward_env
        self.value_buf[self.pointer] = value
        self.logp_buf[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0): # Finish the trajectory by computing advantage estimates and rewards-to-go (完成轨迹，通过计算优势估计和奖励到期)
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buf[path_slice], last_value)
        values = np.append(self.value_buf[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]                           # 计算 TD error
        self.adv_buf[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.GAE_lam)    # 积累 TD error, 得到优势估计（尚未归一化）

        self.return_buf[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1] # 计算奖励到期

        self.trajectory_start_index = self.pointer

    def get(self): # 获取所有缓冲区的数据
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (np.mean(self.adv_buf), np.std(self.adv_buf))
        self.adv_buf = (self.adv_buf - advantage_mean) / advantage_std                  # 归一化优势函数
        return (
            self.obs_buf,
            self.act_buf,
            self.adv_buf,
            self.return_buf,
            self.logp_buf,
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
def train_policy(obs_buf, act_buf, logp_buf, adv_buf): # 训练策略网络
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(logprobabilities(actor(obs_buf), act_buf) - logp_buf) # 新策略下动作的概率与旧策略下相同动作概率的比值的指数形式
        min_advantage = tf.where(adv_buf > 0, (1 + clip_ratio) * adv_buf, (1 - clip_ratio) * adv_buf)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_buf, min_advantage)) 
#         actor_entropy = -tf.reduce_mean(logprobabilities(actor(obs_buf), act_buf))  # 计算策略熵
#         policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_buf, min_advantage)) - actor_entropy_coeff * actor_entropy  # 添加熵项
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logp_buf - logprobabilities(actor(obs_buf), act_buf))
    kl = tf.reduce_sum(kl)
    return kl

@tf.function # Train the value function by regression on mean-squared error
def train_value_function(obs_buf, return_buf): # 训练价值函数
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
#         value_loss = value_loss_coeff*tf.reduce_mean((return_buf - critic(obs_buf)) ** 2)
        value_loss = tf.reduce_mean((return_buf - critic(obs_buf)) ** 2)
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
# # print('=====> both_agent_obs is:', both_agent_obs)
# # print('=====> agent_obs_0 is:', agent_obs_0)
# # print('=====> agent_obs_1 is:', agent_obs_1)
# # print('=====> overcooked_state is:', overcooked_state)
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

buffer = Buffer(observation_dimensions, steps_per_epoch, gamma, GAE_lam) # Initialize the buffer (# 初始化缓冲区)
actor, critic = func_nn_ppo(observation_dimensions, num_actions)

# observation_input = tf.keras.Input(shape=(observation_dimensions,), dtype="float32")
# action_logits_AI_agent = mlp(observation_input, list(mlp_hidden_sizes) + [num_actions])
# actor = tf.keras.Model(inputs=observation_input, outputs=action_logits_AI_agent, name='actor_keras')
# value = tf.squeeze(mlp(observation_input, list(mlp_hidden_sizes) + [1]), axis=1)
# critic = tf.keras.Model(inputs=observation_input, outputs=value, name='critic_keras')

# actor.summary()
# critic.summary()

# Initialize the policy and the value function optimizers
if bc_model_path_train == "./bc_runs_ireca/reproduce_train/cramped_room":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_cr)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_cr)
elif bc_model_path_train == "./bc_runs_ireca/reproduce_train/asymmetric_advantages":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_aa)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_aa)


"""
## Train
"""

# -- 现在你可以使用这些变量进行后续的操作，比如决定哪个智能体行动，或者基于当前状态进行策略计算等
observation_AI = np.array(both_agent_obs[1 - other_agent_env_idx])
observation_HM = np.array(both_agent_obs[other_agent_env_idx])

episode_return_sparse, episode_return_shaped, episode_return_env = 0, 0, 0
episode_return_cau, episode_return_ent, episode_return_ireca = 0, 0, 0
episode_length = 0
count_step = tf.Variable(0, dtype=tf.float32)

avg_return_shaped = []
avg_return_sparse = []
avg_return_env = []
avg_return_cau = []
avg_return_ent = []
avg_return_ireca = []
coeff_reward_softmax_env = []
coeff_reward_softmax_cau = []
coeff_reward_softmax_ent = []
coeff_reward_softmax_env_t = 1.0
coeff_reward_softmax_cau_t = 1.0
coeff_reward_softmax_ent_t = 1.0

# episode_return_sparse_pre = 100
# episode_return_shaped_pre = 100
# episode_return_cau_pre = 0
# episode_return_ent_pre = 0

mean_return_sparse_pre = 100.0
mean_return_shaped_pre = 100.0
mean_return_cau_pre = 0
mean_return_ent_pre = 0


for epoch in range(epochs):

    sum_return_sparse = 0
    sum_return_shaped = 0
    sum_return_cau = 0
    sum_return_ent = 0
    sum_return_env = 0
    sum_return_ireca = 0

    sum_length = 0
    num_episodes = 0

    observation_AI = tf.reshape(observation_AI, (1, -1))
    observation_HM = tf.reshape(observation_HM, (1, -1))

    for t in range(steps_per_epoch):

        # count_step += 1
        count_step.assign_add(1.)

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)
        action_probs_AI_agent = tf.nn.softmax(action_logits_AI_agent)

        action_logits_bc_agent = bc_model_train(observation_HM, training=False)        
        action_HM_agent = tf.argmax(action_logits_bc_agent, axis=1) # 不用随机策略
        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)

        # ----------------------------------------------------------------------------------------------------------------------------
        ########## CALCULATE CAUSAL INFLUENCE ##########

        # ---- 只有 HM 动，看看对 AI 的影响 ----
        obs_dummy_step_know_HM_action = env.dummy_step([4, action_HM_agent.numpy()[0]]) # 让 AI agent 不动
        observation_AI_know_HM_action = tf.reshape(obs_dummy_step_know_HM_action[1-other_agent_env_idx], (1,-1)) # 得到只有 ai agent 行动以后的 HM 的 observation

        action_logis_AI_know_HM_action, _ = sample_action(observation_AI_know_HM_action)
        action_probs_AI_know_HM_action = tf.nn.softmax(action_logis_AI_know_HM_action)

#         reward_cau_AI = tf.keras.losses.KLDivergence()(action_probs_AI_agent, action_probs_AI_know_HM_action)
#         reward_cau_AI = tf.keras.losses.CategoricalCrossentropy()(action_probs_AI_agent, action_probs_AI_know_HM_action)
#         reward_cau_AI = tf.reduce_mean(tf.math.log(action_probs_AI_agent/(action_probs_AI_know_HM_action+1e-12)), axis=1)[0]
        reward_cau_AI = tf.reduce_mean(tf.math.abs(tf.math.log(action_probs_AI_know_HM_action/(action_probs_AI_agent+1e-12))), axis=1)[0]

        # ---- 只有 AI 动，看看对 HM 的影响 ----
#         obs_dummy_step_know_AI_action = env.dummy_step([action_AI_agent.numpy()[0], 4]) # 让 HM agent 不动
#         observation_HM_know_AI_action = tf.reshape(obs_dummy_step_know_AI_action[other_agent_env_idx], (1,-1)) # 得到只有 ai agent 行动以后的 HM 的 observation

#         action_logis_HM_know_AI_action = bc_model(observation_HM_know_AI_action, training=False)
#         action_probs_HM_know_AI_action = tf.nn.softmax(action_logis_HM_know_AI_action)

# #         reward_cau_HM = tf.keras.losses.KLDivergence()(action_probs_bc_agent, action_probs_HM_know_AI_action)
#         # reward_cau_HM = tf.keras.losses.CategoricalCrossentropy()(action_probs_bc_agent, action_probs_HM_know_AI_action)
#         # reward_cau_HM = tf.reduce_mean(tf.math.log(action_probs_HM_know_AI_action/(action_probs_bc_agent+1e-12)), axis=1)[0]
#         reward_cau_HM = tf.reduce_mean(tf.math.abs(tf.math.log(action_probs_HM_know_AI_action/(action_probs_bc_agent+1e-12))), axis=1)[0]        

        reward_cau = coeff_reward_cau*reward_cau_AI
#         reward_cau = reward_cau_HM

        # ----------------------------------------------------------------------------------------------------------------------------
        
        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step([action_AI_agent.numpy()[0], action_HM_agent.numpy()[0]])

        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1-other_agent_env_idx], (1,-1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1,-1))

        coeff_reward_shaped = max(0.0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped*reward_shaped

        # -------------------------------------
        # reward_ent = tfp.distributions.Categorical(action_probs_AI_agent).entropy()[0]
        reward_ent = -tf.reduce_mean(tf.math.log(action_probs_AI_agent+1e-12))
        reward_ent = coeff_reward_ent*reward_ent

        # ----------------------------
#         reward_cau = tf.minimum(reward_env, reward_cau) # 相当于在同时没有 sparse 和 shaped 的时候，置零
        # if reward_sparse > 0:
        #     reward_cau = 20*reward_cau

        # reward_ireca = reward_env + reward_cau + reward_ent
        # reward_ireca = reward_env + reward_intrinsic

        reward_ireca = coeff_reward_softmax_env_t*reward_env + coeff_reward_softmax_cau_t*reward_cau + coeff_reward_softmax_ent_t*reward_ent

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped        
        episode_return_cau += reward_cau
        episode_return_ent += reward_ent
        episode_return_env += reward_env
        episode_return_ireca += reward_ireca
        episode_length += 1
        
        # Get the value and log-probability of the action_AI_agent (获取动作的价值和对数概率)
        value_t = critic(observation_AI)
        logprobability_t = logprobabilities(action_logits_AI_agent, action_AI_agent)

        # Store obs, act, rew, v_t, logp_pi_t (存储观测、动作、奖励、价值和对数概率)
        buffer.store(observation_AI, action_AI_agent, reward_ireca, value_t, logprobability_t)

        # Update the observation_AI (更新观测)
        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

        # Finish trajectory if reached to a terminal state (如果达到终止状态则结束轨迹)
        # terminal = done
        if done or (t == steps_per_epoch - 1):

            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            buffer.finish_trajectory(last_value)

            sum_return_sparse += episode_return_sparse
            sum_return_shaped += episode_return_shaped
            sum_return_cau += episode_return_cau
            sum_return_ent += episode_return_ent
            sum_return_env += episode_return_env
            sum_return_ireca += episode_return_ireca
            sum_length += episode_length
            num_episodes += 1
            obs_dict = env.reset()
            observation_AI = tf.reshape(obs_dict["both_agent_obs"][1-other_agent_env_idx], (1,-1))
            observation_HM = tf.reshape(obs_dict["both_agent_obs"][other_agent_env_idx], (1,-1))

            # ---- (begin) 同一个 policy 用不同的 coeff_reward_softmax 系数 （每个 episode 都换系数）----

            # episode_return_env_delta = tf.math.abs(episode_return_sparse + episode_return_shaped - episode_return_sparse_pre - episode_return_shaped_pre)
            # episode_return_cau_delta = tf.math.abs(episode_return_cau - episode_return_cau_pre)
            # episode_return_ent_delta = tf.math.abs(episode_return_ent - episode_return_ent_pre)

            # coeff_reward_softmax = tf.nn.softmax([episode_return_env_delta, episode_return_cau_delta, episode_return_ent_delta])
            
            # episode_return_sparse_pre = episode_return_sparse
            # episode_return_shaped_pre = episode_return_shaped
            # episode_return_cau_pre = episode_return_cau
            # episode_return_ent_pre = episode_return_ent
            # print(f"                                     COE softmax: {coeff_reward_softmax[0].numpy()}, {coeff_reward_softmax[1].numpy()}, {coeff_reward_softmax[2].numpy()}")

            # ---- (end) 同一个 policy 用不同的 coeff_reward_softmax 系数 ----

            episode_return_sparse, episode_return_shaped, episode_return_cau, episode_return_ent = 0, 0, 0, 0
            episode_return_env, episode_return_ireca = 0, 0
            episode_length = 0

    mean_length        = sum_length / num_episodes
    mean_return_sparse = sum_return_sparse / num_episodes
    mean_return_shaped = sum_return_shaped / num_episodes
    mean_return_cau    = sum_return_cau / num_episodes
    mean_return_ent    = sum_return_ent / num_episodes #min(mean_return_ent_max, sum_return_ent / num_episodes)
    mean_return_env    = sum_return_env / num_episodes
    mean_return_ireca   = sum_return_ireca / num_episodes

    # ---- (begin) 同一个 policy 用相同的 coeff_reward_softmax 系数 ----

    if epoch < max_ireca_epoch:
        #  如果学习变差了（reward变小了），则要重点调整 (所以，如果下面的 delta 越大，则是越差)
        mean_return_env_delta = tf.math.divide(mean_return_sparse_pre + mean_return_shaped_pre, mean_return_sparse + mean_return_shaped+1e-12)
        mean_return_cau_delta = tf.math.divide(mean_return_cau_pre, mean_return_cau+1e-12)
        mean_return_ent_delta = tf.math.divide(mean_return_ent_pre, mean_return_ent+1e-12)
        # mean_return_env_delta = tf.math.subtract(mean_return_sparse_pre + mean_return_shaped_pre, mean_return_sparse + mean_return_shaped+1e-12)
        # mean_return_cau_delta = tf.math.subtract(mean_return_cau_pre, mean_return_cau)
        # mean_return_ent_delta = tf.math.subtract(mean_return_ent_pre, mean_return_ent)
        # 
        coeff_reward_softmax_normilized = tf.nn.softmax([mean_return_env_delta, mean_return_cau_delta, mean_return_ent_delta])
        coeff_reward_softmax = 3.0 * coeff_reward_softmax_normilized
        # 
        coeff_reward_softmax_env_t = coeff_reward_softmax[0]
        coeff_reward_softmax_cau_t = coeff_reward_softmax[1]
        coeff_reward_softmax_ent_t = coeff_reward_softmax[2]
    else:
        coeff_reward_softmax_env_t = 1.0
        coeff_reward_softmax_cau_t = 0.0
        coeff_reward_softmax_ent_t = 0.0


    # mean_return_env_pre = mean_return_env
    mean_return_sparse_pre = mean_return_sparse
    mean_return_shaped_pre = mean_return_shaped
    mean_return_cau_pre = mean_return_cau
    mean_return_ent_pre = mean_return_ent

    # ---- (end) 同一个 policy 用相同的 coeff_reward_softmax 系数 ----

    (obs_buf, act_buf, adv_buf, ret_buf, logp_buf) = buffer.get()


    for _ in range(iterations_train_policy):
        for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size):
            kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
            if kl > 1.5 * target_kl:
                break
            train_value_function(obs_batch, ret_batch)

    print(f" [ireca] Epoch: ({epoch}). Mean Length: ({mean_length}).")
    print(f" Mean sparse: {mean_return_sparse}. Mean shaped: {mean_return_shaped}. Mean cau: {mean_return_cau}. Mean entropy: {mean_return_ent}.")
    print(f"                   COE softmax: {coeff_reward_softmax_env_t}, {coeff_reward_softmax_cau_t}, {coeff_reward_softmax_ent_t}.")
    print(f"                   Mean Env: {mean_return_env}. Mean ireca: {mean_return_ireca}.")

    avg_return_sparse.append(mean_return_sparse)
    avg_return_shaped.append(mean_return_shaped)
    avg_return_cau.append(mean_return_cau)
    avg_return_ent.append(mean_return_ent)
    avg_return_env.append(mean_return_env)
    avg_return_ireca.append(mean_return_ireca)
    coeff_reward_softmax_env.append(coeff_reward_softmax_env_t)
    coeff_reward_softmax_cau.append(coeff_reward_softmax_cau_t)
    coeff_reward_softmax_ent.append(coeff_reward_softmax_ent_t)

    if ((epoch+1) % 20 == 0) or ((epoch+1) == epochs) :
        np.save('./data_tmp/data_ireca_return_sparse.npy',  avg_return_sparse)
        np.save('./data_tmp/data_ireca_return_shaped.npy',  avg_return_shaped)
        np.save('./data_tmp/data_ireca_return_cau.npy',     avg_return_cau)
        np.save('./data_tmp/data_ireca_return_ent.npy',     avg_return_ent)
        np.save('./data_tmp/data_ireca_return_env.npy',     avg_return_env)
        np.save('./data_tmp/data_ireca_return_ireca.npy',    avg_return_ireca)


if bc_model_path_train == "./bc_runs_ccima/reproduce_train/cramped_room":
    actor.save_weights("model_cr_actor_ireca.h5")
    critic.save_weights("model_cr_critic_ireca.h5")
elif bc_model_path_train == "./bc_runs_ccima/reproduce_train/asymmetric_advantages":
    actor.save_weights("model_aa_actor_ireca.h5")
    critic.save_weights("model_aa_critic_ireca.h5")

    
# # ----Plot Figure----
# plt.figure()
# plt.plot(avg_return_env, markerfacecolor='none')
# # plt.xlabel('Index of data samples')
# plt.ylabel('avg_return_env')
# plt.legend(fontsize=12, loc='lower right')
# plt.savefig('./figs/ireca_avg_return_env.pdf', format='pdf')  

# plt.show()