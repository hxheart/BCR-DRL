epochs = int(400)
horizon_len = int(400)

horizon_num = int(5)

steps_per_epoch = int(horizon_num*horizon_len) # agents 和 env 交互的次数 (是先和 env 交互这么多次，然后再出现采样 mini-batch 组 data 来训练)

batch_size = int(steps_per_epoch/2)


iterations_train_policy = int(60)	# 在每个 episode 中，policy 网络更新的次数；策略网络决定了代理在每一步采取的行动
# iterations_train_value = int(60)		# 表示我们在每个episode中会更新价值（value）网络的次数。价值网络预测每个状态的长期回报

# -- PPO -
# mean_return_cau_max = 30.0
# mean_return_ent_max = 30.0

coeff_reward_cau = 1
coeff_reward_ent = 0.03

# actor_entropy_coeff= 0   # actor 部分，熵正则化系数，控制熵在policy损失中的权重
# value_loss_coeff = 1

gamma = 0.9		# 衡量未来奖励的重要性：越接近 1，越重视长期回报
clip_ratio = 0.03 	# 限制策略更新幅度：较小的值可以防止策略更新过大
learning_rate_policy_cr = 1e-4
learning_rate_value_cr = 3e-4 	# 用于调整价值函数估计的精度；通常比 policy_lr 大，因为 value 估计更倾向追求精度，而非探索

GAE_lam = 0.9			# Generalized Advantage Estimation 中的衰减因子：越大的值，表示，计算优势时，近期回报差异比远期的更加重要
target_kl = 0.01 	# 目标 KL 散度：控制每次更新时，策略分布相较旧策略的变化程度。较小的值，可以维持学习过程的稳定
 
learning_rate_reward_shaping = 4e-6 # 1/(100*5*400)=1/(2e5)=5e-6 # 4e-6=1/(2.5e5)

# ----

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
import gym
# ----
bc_model_path_train = "./bc_runs_ireca/reproduce_train/cramped_room"
bc_model_path_test  = "./bc_runs_ireca/reproduce_test/cramped_room"
mdp = OvercookedGridworld.from_layout_name("cramped_room")




# ----
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon_len)
env = gym.make("Overcooked-v0", base_env = base_env, featurize_fn = base_env.featurize_state_mdp)

# -------- (BC human) model loading --------
from BC_model_functions import load_bc_model
# --
bc_model_train, bc_params_train = load_bc_model(bc_model_path_train)
bc_model_test,  bc_params_test  = load_bc_model(bc_model_path_test)


# -----------------------
if bc_model_path_train == "./bc_runs_ireca/reproduce_train/cramped_room":
    max_ireca_epoch = 100
    
    
    
    
    
    
    
    
    
