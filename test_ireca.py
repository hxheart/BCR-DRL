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


def discounted_cumulative_sums(x, discount): #  Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates (计算折扣累计和，用于计算奖励到期和优势估计)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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

actor, critic = func_nn_ppo(observation_dimensions, num_actions)



if bc_model_path_test == "./bc_runs_ireca/reproduce_test/cramped_room":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_cr)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_cr)
    actor.load_weights("./data/model/model_cr_actor_ireca.h5")
    critic.load_weights("./data/model/model_cr_critic_ireca.h5")
    print("loaded the cramped_room weights")
elif bc_model_path_test == "./bc_runs_ireca/reproduce_test/asymmetric_advantages":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_aa)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_aa)
    actor.load_weights("./data/model/model_aa_actor_ireca.h5")
    critic.load_weights("./data/model/model_aa_critic_ireca.h5")
    print("\n loaded the asymmetric_advantages weights \n")
    
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

        count_step.assign_add(1.)

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)
        action_probs_AI_agent = tf.nn.softmax(action_logits_AI_agent)

        action_logits_bc_agent = bc_model_test(observation_HM, training=False)        
        action_HM_agent = tf.argmax(action_logits_bc_agent, axis=1) # 不用随机策略
        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)

        # ----------------------------------------------------------------------------------------------------------------------------
        ########## CALCULATE CAUSAL INFLUENCE ##########

        obs_dummy_step_know_HM_action = env.dummy_step([4, action_HM_agent.numpy()[0]]) # 让 AI agent 不动
        observation_AI_know_HM_action = tf.reshape(obs_dummy_step_know_HM_action[1-other_agent_env_idx], (1,-1)) # 得到只有 ai agent 行动以后的 HM 的 observation

        action_logis_AI_know_HM_action, _ = sample_action(observation_AI_know_HM_action)
        action_probs_AI_know_HM_action = tf.nn.softmax(action_logis_AI_know_HM_action)

        reward_cau_AI = tf.reduce_mean(tf.math.abs(tf.math.log(action_probs_AI_know_HM_action/(action_probs_AI_agent+1e-12))), axis=1)[0]

        reward_cau = coeff_reward_cau*reward_cau_AI

        # ----------------------------------------------------------------------------------------------------------------------------
        
        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step([action_AI_agent.numpy()[0], action_HM_agent.numpy()[0]])

        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1-other_agent_env_idx], (1,-1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1,-1))

#         coeff_reward_shaped = max(0.0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse# + coeff_reward_shaped*reward_shaped

        # -------------------------------------
        reward_ent = -tf.reduce_mean(tf.math.log(action_probs_AI_agent+1e-12))
        reward_ent = coeff_reward_ent*reward_ent

        # ----------------------------
        reward_ireca = coeff_reward_softmax_env_t*reward_env + coeff_reward_softmax_cau_t*reward_cau + coeff_reward_softmax_ent_t*reward_ent

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped        
        episode_return_cau += reward_cau
        episode_return_ent += reward_ent
        episode_return_env += reward_env
        episode_return_ireca += reward_ireca
        episode_length += 1

        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

        if done or (t == steps_per_epoch - 1):

            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            # buffer.finish_trajectory(last_value)

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
    coeff_reward_softmax_env_t = 1.0
    coeff_reward_softmax_cau_t = 0.0
    coeff_reward_softmax_ent_t = 0.0


    # mean_return_env_pre = mean_return_env
    mean_return_sparse_pre = mean_return_sparse
    mean_return_shaped_pre = mean_return_shaped
    mean_return_cau_pre = mean_return_cau
    mean_return_ent_pre = mean_return_ent

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

