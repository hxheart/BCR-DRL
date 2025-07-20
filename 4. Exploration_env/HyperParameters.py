import tensorflow as tf
import numpy as np


epochs = 500
steps_per_epoch = 400
batch_size = int(steps_per_epoch/4)
target_kl = 0.01    # 目标 KL 散度：控制每次更新时，策略分布相较旧策略的变化程度。较小的值，可以维持学习过程的稳定
    
max_caci_epoch = steps_per_epoch*0.5
coeff_reward_cau = 0.1
coeff_reward_ent = 0.01

iterations_train_policy = 30
clip_ratio= 0.05
learning_rate_policy = 3e-4
learning_rate_value = 1e-3


    
GRID_SIZE = 4
UNIT = 36
    
from env_Exploration import Explore
env = Explore(render_mode=True) # 确保启用了渲染
# env = Explore()
num_actions = 5
observation_dimensions = GRID_SIZE*GRID_SIZE+4   # env.observation_space.shape[0]

both_agent_obs = env.reset() # 得到 reset 函数返回的字典值
observation_AI = np.array(both_agent_obs)
episode_return_sparse = 0
episode_return_env, episode_length = 0, 0
count_step = 0


