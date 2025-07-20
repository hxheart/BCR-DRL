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

def discounted_cumulative_sums(x, discount): #  Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates (计算折扣累计和，用于计算奖励到期和优势估计)
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def logprobabilities(action_logits_AI_agent, a): # Compute the log-probabilities of taking actions a by using the action_logits_AI_agent (i.e. the output of the actor) (计算动作 a 的对数概率)
    logprobabilities_all = tf.nn.log_softmax(action_logits_AI_agent)
    logprobability = tf.reduce_sum(tf.one_hot(a, num_actions) * logprobabilities_all, axis=1)
    return logprobability

@tf.function # Sample action_AI_agent from actor
def sample_action(observation_AI):
    action_logits_AI_agent = actor(observation_AI)
    action_AI_agent = tf.squeeze(tf.random.categorical(action_logits_AI_agent, 1, seed=seed_generator), axis=1)
    return action_logits_AI_agent, action_AI_agent

"""
## Initializations  
"""

observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n


obs_dict = env.reset() # 得到 reset 函数返回的字典值
# -- 把字典里面的各种物理量提取出来 --
both_agent_obs = obs_dict["both_agent_obs"] # 获取两个智能体的观察值
agent_obs_0 = both_agent_obs[0]  # 第一个智能体的观察值
agent_obs_1 = both_agent_obs[1]  # 第二个智能体的观察值
overcooked_state = obs_dict["overcooked_state"] # 获取当前的Overcooked状态
other_agent_env_idx = obs_dict["other_agent_env_idx"] # 获取另一个智能体的环境索引


# buffer = Buffer(observation_dimensions, steps_per_epoch) # Initialize the buffer (# 初始化缓冲区)
actor, critic = func_nn_ppo(observation_dimensions, num_actions)

if bc_model_path_test == "./bc_runs_ireca/reproduce_test/cramped_room":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_cr)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_cr)
    actor.load_weights("./data/model/model_cr_actor_causal.h5")
    critic.load_weights("./data/model/model_cr_critic_causal.h5")
    print('\n ----> LOAD cramped_room weights \n ')


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
        action_logits_bc_agent = bc_model_test(observation_HM, training=False)        

        # ----------------------------------------------------------------------------------------------------------------------------
        ########## CALCULATE CAUSAL INFLUENCE ##########
        # ---- KL 散度公式的左边，知道 AI agent action 以后 --
        obs_both_agent_know_AI_action = env.dummy_step([action_AI_agent.numpy()[0], 4]) # 让 HM agent 不动
        observation_HM_know_AI_action = tf.reshape(obs_both_agent_know_AI_action[other_agent_env_idx], (1,-1)) # 得到只有 ai agent 行动以后的 HM 的 observation

        action_logits_HM_know_AI_action = bc_model_test(observation_HM_know_AI_action, training=False)
        action_probs_HM_know_AI_action = tf.nn.softmax(action_logits_HM_know_AI_action) # 这个其实就是边缘概率分布

        # ---- KL 散度公式的右边，不知道 AI agent action，其实就是用一开始的状态估计出来的 distribution  --
        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)

        # - 计算 CiMAR 的 KL divergence
        reward_cau = tf.keras.losses.KLDivergence()(action_probs_HM_know_AI_action, action_probs_bc_agent)

        # ----------------------------------------------------------------------------------------------------------------------------
        
        action_HM_agent = tf.argmax(action_logits_bc_agent, axis=1)

        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step([action_AI_agent.numpy()[0], action_HM_agent.numpy()[0]])

        coeff_reward_shaped = max(0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse + coeff_reward_shaped*reward_shaped

        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1-other_agent_env_idx], (1,-1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1,-1))

#         coeff_reward_shaped = max(0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse# + coeff_reward_shaped*reward_shaped

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
        # buffer.store(observation_AI, action_AI_agent, reward_env, value_t, logprobability_t)

        # Update the observation_AI (更新观测)
        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

        # Finish trajectory if reached to a terminal state (如果达到终止状态则结束轨迹)
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            # print('count_step:', count_step)
            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            # buffer.finish_trajectory(last_value)
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


    print(f" [causal] Epoch: {epoch}. Mean Length: {sum_length / num_episodes}")
    print(f" Mean sparse: {sum_return_sparse / num_episodes}. Mean shaped: {sum_return_shaped / num_episodes}. Mean Env: {sum_return_env / num_episodes}. Mean cau: {sum_return_cau / num_episodes}. Mean causal: {sum_return_causal / num_episodes}.")
    # print('num_episodes:', num_episodes)

    avg_return_shaped.append(sum_return_shaped / num_episodes)
    avg_return_sparse.append(sum_return_sparse / num_episodes)
    avg_return_env.append(sum_return_env / num_episodes)
    avg_return_cau.append(sum_return_cau / num_episodes)
    avg_return_causal.append(sum_return_causal / num_episodes)




