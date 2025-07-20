import os
os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用哪块GPU

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

# @tf.function是TensorFlow库中的一个装饰器，用于将普通的Python函数转换为TensorFlow图形函数(也称为计算图函数)。它的主要作用和目的是为了加速执行，
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
both_agent_obs = obs_dict["both_agent_obs"] # 获取两个智能体的观察值
other_agent_env_idx = obs_dict["other_agent_env_idx"] # 获取另一个智能体的环境索引



# buffer = Buffer(observation_dimensions, steps_per_epoch) # Initialize the buffer (# 初始化缓冲区)
actor, critic = func_nn_ppo(observation_dimensions, num_actions)

if bc_model_path_test == "./bc_runs_ireca/reproduce_test/asymmetric_advantages":
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy_aa)
    value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value_aa)
    actor.load_weights("./data/model/model_aa_actor_ppobc.h5")
    critic.load_weights("./data/model/model_aa_critic_ppobc.h5")
    print("\n ----> loaded the asymmetric_advantages weights \n")

    
# -- 现在你可以使用这些变量进行后续的操作，比如决定哪个智能体行动，或者基于当前状态进行策略计算等
observation_AI = np.array(both_agent_obs[1 - other_agent_env_idx])
observation_HM = np.array(both_agent_obs[other_agent_env_idx])

episode_return_sparse, episode_return_shaped = 0, 0
episode_return_env, episode_length = 0, 0
count_step = 0



"""
## Train
"""

avg_return_shaped = []
avg_return_sparse = []
avg_return_env = []

for epoch in range(epochs):

    sum_return_sparse = 0
    sum_return_shaped = 0
    sum_return_env = 0
    sum_length = 0
    num_episodes = 0

    observation_AI = tf.reshape(observation_AI, (1, -1))
    observation_HM = tf.reshape(observation_HM, (1, -1))

    for t in range(steps_per_epoch):

        count_step += 1

        action_logits_AI_agent, action_AI_agent = sample_action(observation_AI)

        action_logits_bc_agent = bc_model_test(observation_HM, training=False)
        
        action_probs_bc_agent = tf.nn.softmax(action_logits_bc_agent)
        action_HM_agent = tf.argmax(action_probs_bc_agent, axis=1) # 不用随机策略

        # - Convert TensorFlow tensors to Python integers
        action_AI_agent_np = action_AI_agent.numpy().tolist()[0]
        action_HM_agent_np = action_HM_agent.numpy().tolist()[0]
        action_np = [action_AI_agent_np, action_HM_agent_np]
        
        # observation_AI_new, reward_env, done, _, _ = env.step(action_AI_agent[0].numpy())
        obs_dict_new, reward_sparse, reward_shaped, done, _ = env.step(action_np)
        
        observation_AI_new = tf.reshape(obs_dict_new["both_agent_obs"][1 - other_agent_env_idx], (1,-1))
        observation_HM_new = tf.reshape(obs_dict_new["both_agent_obs"][other_agent_env_idx], (1,-1))

#         coeff_reward_shaped = max(0, 1-count_step*learning_rate_reward_shaping)
        reward_env = reward_sparse# + coeff_reward_shaped*reward_shaped
#         reward_env = tf.add(reward_sparse, tf.multiply(coeff_reward_shaped, reward_shaped))

        episode_return_sparse += reward_sparse
        episode_return_shaped += reward_shaped        
        episode_return_env += reward_env
        episode_length += 1

        # Get the value and log-probability of the action_AI_agent (获取动作的价值和对数概率)
#         value_t = critic(observation_AI)
#         logprobability_t = logprobabilities(action_logits_AI_agent, action_AI_agent)

        # Store obs, act, rew, v_t, logp_pi_t (存储观测、动作、奖励、价值和对数概率)
        # buffer.store(observation_AI, action_AI_agent, reward_env, value_t, logprobability_t)

        # Update the observation_AI (更新观测)
        observation_AI = observation_AI_new
        observation_HM = observation_HM_new

        # Finish trajectory if reached to a terminal state (如果达到终止状态则结束轨迹)
#         terminal = done
        if done or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation_AI.reshape(1, -1))
            # buffer.finish_trajectory(last_value)
            sum_return_shaped += episode_return_shaped
            sum_return_sparse += episode_return_sparse
            sum_return_env += episode_return_env
            sum_length += episode_length
            num_episodes += 1
            obs_dict = env.reset()
            observation_AI = tf.reshape(obs_dict["both_agent_obs"][1-other_agent_env_idx], (1,-1))
            observation_HM = tf.reshape(obs_dict["both_agent_obs"][other_agent_env_idx], (1,-1))
            episode_return_shaped, episode_return_sparse, episode_return_env, episode_length = 0, 0, 0, 0

    # (obs_buf, act_buf, adv_buf, ret_buf, logp_buf) = buffer.get()

    # for _ in range(iterations_train_policy):
    #     for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in tf_get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size):
    #     # for obs_batch, act_batch, adv_batch, ret_batch, logp_batch in get_mini_batches(obs_buf, act_buf, adv_buf, ret_buf, logp_buf, batch_size=batch_size):
    #         kl = train_policy(obs_batch, act_batch, logp_batch, adv_batch)
    #         if kl > 1.5 * target_kl:
    #             break 
    #         train_value_function(obs_batch, ret_batch)


    print(f" [ppobc] Epoch: {epoch}. Mean Length: {sum_length / num_episodes}")
    print(f" Mean sparse: {sum_return_sparse / num_episodes}. Mean shaped: {sum_return_shaped / num_episodes}. Mean Env: {sum_return_env / num_episodes}. ")

    avg_return_shaped.append(sum_return_shaped / num_episodes)
    avg_return_sparse.append(sum_return_sparse / num_episodes)
    # avg_return_env.append(sum_return_env / num_episodes)
    


