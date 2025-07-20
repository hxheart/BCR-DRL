epochs = int(600)
horizon_len = int(400)

horizon_num = int(5)
steps_per_epoch = int(horizon_num*horizon_len) # agents 和 env 交互的次数 (是先和 env 交互这么多次，然后再出现采样 mini-batch 组 data 来训练)
batch_size = int(steps_per_epoch/2)


# -- PPO -
# mean_return_cau_max = 30.0
# mean_return_ent_max = 30.0

target_kl = 0.01    # 目标 KL 散度：控制每次更新时，策略分布相较旧策略的变化程度。较小的值，可以维持学习过程的稳定

# actor_entropy_coeff= 0   # actor 部分，熵正则化系数，控制熵在policy损失中的权重
# value_loss_coeff = 1


from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
import gym


# # ------------------------------------------------------------------------------
coeff_reward_cau_co = 0.2
coeff_reward_ent_co = 0.01
gamma_co = 0.99
GAE_lam_co = 0.98
clip_ratio_co = 0.02
learning_rate_policy_co = 1e-4
learning_rate_value_co = 2e-4
learning_rate_reward_shaping_co = 2e-6
iterations_train_policy_co = int(60)   # 在每个 episode 中，policy 网络更新的次数；策略网络决定了代理在每一步采取的行动
iterations_train_value_co = int(60)      # 表示我们在每个episode中会更新价值（value）网络的次数。价值网络预测每个状态的长期回报
# ----
bc_model_path_train = "./bc_runs_ccima/reproduce_train/coordination_ring"
bc_model_path_test  = "./bc_runs_ccima/reproduce_test/coordination_ring"
bc_model_path_test = "./bc_runs_ccima/reproduce_train/coordination_ring"
mdp = OvercookedGridworld.from_layout_name("coordination_ring")

# ----
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon_len)
env = gym.make("Overcooked-v0", base_env = base_env, featurize_fn = base_env.featurize_state_mdp)

# -------- (BC human) model loading --------
from BC_model_functions import load_bc_model
# --
bc_model_train, bc_params_train = load_bc_model(bc_model_path_train)
bc_model_test,  bc_params_test  = load_bc_model(bc_model_path_test)


# -----------------------
if bc_model_path_train == "./bc_runs_ccima/reproduce_train/coordination_ring":
    max_caci_epoch = 300
    
    
    
    
    
    
    
    
    
