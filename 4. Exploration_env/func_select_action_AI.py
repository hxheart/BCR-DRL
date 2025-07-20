from HyperParameters import *
import tensorflow as tf
import numpy as np


class OUNoise:
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.05):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state
    
    def get_action_noise(self):
        ou_state = self.evolve_state()
        return ou_state
    

# random action selection 主策略
def func_select_action_AI(num_actions, count_episode, action_probs_AI_agent, epsilon_AI_action):
    
    ou_noise = OUNoise(action_dimension=num_actions)

    if count_episode < AI_action_episodes_random:
        action_AI_agent = tf.random.uniform([1], 0, num_actions, dtype=tf.int64)
    
    elif count_episode < AI_action_episodes_epsilon_greedy:
        epsilon_AI_action -= learning_rate_epsilon_AI
        epsilon_AI_action = max(epsilon_AI_action_min, epsilon_AI_action)
        if np.random.rand() < epsilon_AI_action:
            action_AI_agent = tf.random.uniform([1], 0, num_actions, dtype=tf.int64)
        else:
            action_ou_noise = ou_noise.get_action_noise() # 获取OU噪声
            action_probs_noisy = action_probs_AI_agent + action_ou_noise # 给动作概率添加噪声
            # action_probs_clip = np.clip(action_probs_noisy, 0, 1) # 确保动作概率在合理范围内
            action_argmax = tf.argmax(action_probs_noisy, axis=1).numpy() # 选择动作
            action_AI_agent = tf.convert_to_tensor([action_argmax[0]], dtype=tf.int64) # 转换成张量
    else:
        action_ou_noise = ou_noise.get_action_noise()  # 获取OU噪声
        action_probs_noisy = action_probs_AI_agent + action_ou_noise  # 给动作概率添加噪声
        # action_probs_clip = np.clip(action_probs_noisy, 0, 1)  # 确保动作概率在合理范围内
        action_AI_agent = tf.argmax(action_probs_noisy, axis=1)  # 选择动作
        # action_AI_agent = tf.convert_to_tensor([action_argmax[0]], dtype=tf.int64)  # 转换成张量
    
    return action_AI_agent