# 该脚本通过导入必要的模块和设置环境变量，使用 sacred 进行实验管理，并配置 Slack 通知
# 通过定义环境创建函数和配置运行函数，脚本能够训练 PPO 自学习代理。注释和临时文档提供了使用该脚本的说明

# All imports except rllib
import argparse    # argparse 用于解析命令行参数。
import os          
import sys         # os 和 sys 用于操作系统相关功能。
import warnings    # warnings 用于控制警告信息。

import numpy as np

from overcooked_ai_py.agents.benchmarking import AgentEvaluator # AgentEvaluator 用于评估代理的性能。(在tutorial的第三步里，首先就是要create an AgentEvaluator)

warnings.simplefilter("ignore") # 忽略所有警告信息。

# environment variable that tells us whether this code is running on the server or not
LOCAL_TESTING = os.getenv("RUN_ENV", "production") == "local" # 检查环境变量 RUN_ENV 是否设置为 local，用于区分本地和服务器环境。

# Sacred setup (must be before rllib imports)
from sacred import Experiment 

ex = Experiment("PPO RLLib") # 使用 sacred 创建一个新的实验，用于记录和管理实验配置和结果。

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS # sacred是一个 Python 库, 可以帮助研究人员配置、组织、记录和复制实验

SETTINGS.CONFIG.READ_ONLY_CONFIG = False # 修改 sacred 设置，使其配置可写，以便与 rllib 兼容。

# Slack notification configuration
from sacred.observers import SlackObserver

if os.path.exists("slack.json") and not LOCAL_TESTING: # 如果存在 slack.json 文件且不是本地环境，则配置 Slack 通知
    slack_obs = SlackObserver.from_config("slack.json") # SlackObserver 用于在实验运行时发送 Slack 通知。
    ex.observers.append(slack_obs)

    # Necessary for capturing stdout in multiprocessing setting
    SETTINGS.CAPTURE_MODE = "sys" # SETTINGS.CAPTURE_MODE = "sys" 用于在多进程设置中捕获标准输出。


# -------- 导入 rllib 和相关依赖模块，用于训练 PPO 模型 --------
# rllib and rllib-dependent imports 
# Note: tensorflow and tensorflow dependent imports must also come after rllib imports
# This is because rllib disables eager execution. Otherwise, it must be manually disabled
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.result import DEFAULT_RESULTS_DIR 

# -------- 导入 human_aware_rl 模块中的函数和类，辅助进行模型训练和保存 --------
from human_aware_rl.imitation.behavior_cloning_tf2 import (
    BC_SAVE_DIR,
    BehaviorCloningPolicy,
)
from human_aware_rl.ppo.ppo_rllib import RllibLSTMPPOModel, RllibPPOModel
from human_aware_rl.rllib.rllib import (
    OvercookedMultiAgent,
    gen_trainer_from_params,
    save_trainer,
)
from human_aware_rl.utils import WANDB_PROJECT

###################### Temp Documentation ####################### 临时文档，说明如何使用该脚本训练 PPO 自学习代理，并查看训练结果。
#   run the following command in order to train a PPO self-play #
#   agent with the static parameters listed in my_config        #
#                                                               #
#   python ppo_rllib_client.py                                  #
#                                                               #
#   In order to view the results of training, run the following #
#   command                                                     #
#                                                               #
#   tensorboard --log-dir ~/ray_results/                        #
#                                                               #
#################################################################

# Dummy wrapper to pass rllib type checks (创建环境的函数包装器，确保环境可以通过 rllib 的类型检查)
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent # 重新导入 OvercookedMultiAgent 以处理序列化问题。 OvercookedMultiAgent是一个Class, 用来 wrap OvercookedEnv in an Rllib compatible multi-agent environment

    return OvercookedMultiAgent.from_config(env_config)


# -------- 下面三个函数的主要逻辑关系 --------
# my_config 函数用于定义实验的默认配置，这些配置会传递给 run 函数
# main 函数是程序的入口，当脚本运行时，main 函数会被调用
# main 函数是程序入口的原因是 @ex.automain 装饰器。 @ex.automain 装饰器来自 sacred 库，专门用于标记一个函数为实验的主入口
# main 函数调用 run 函数，并将 params 作为参数传递给 run 函数
# run 函数使用传递的 params 执行训练过程

# --------- 下面三个函数的代码阅读思路 --------
# @ex.config 装饰器：   定义实验的配置。首先要理解 sacred 如何使用 @ex.config 装饰器来设置实验的配置参数。通常这里会包含超参数、文件路径等信息
# my_config 函数：      阅读并理解 my_config 函数中设置的默认配置。这些配置将会被传递给 run 函数
# run 函数：            理解 run 函数的逻辑。这个函数是训练过程的核心，里面包含了创建环境、配置 PPO 训练参数、训练模型和保存结果的代码
# @ex.automain 装饰器： 理解 @ex.automain 装饰器的作用。这个装饰器告诉 sacred，当脚本运行时，应该执行 main 函数
# main 函数：           理解 main 函数的逻辑。这个函数是程序的入口，会调用 run 函数，并将实验的配置参数传递给 run 函数


# @ex.config 装饰器用于定义实验的默认配置，是 sacred 中的装饰器
# 所有在 @ex.config 装饰器下定义的变量，都会自动被传递给实验的其他部分
@ex.config # 
def my_config(): # 配置 PPO 训练的静态参数。 my_config 函数用于定义实验的默认配置，这些配置会传递给 run 函数。
    ### Resume chekpoint_path (检查点路径，用于恢复训练) ###
    resume_checkpoint_path = None

    ### Model params ###

    use_phi = True # Whether dense reward should come from potential function or not (是否使用潜在函数生成密集奖励)

    # whether to use recurrence in ppo model
    use_lstm = False # 是否在 PPO 模型中使用循环神经网络（LSTM）

    # Base model params (基础模型参数)
    NUM_HIDDEN_LAYERS = 3   # 隐藏层数量
    SIZE_HIDDEN_LAYERS = 64 # 隐藏层大小
    NUM_FILTERS = 256       # 卷积层过滤器数量
    NUM_CONV_LAYERS = 3     # 卷积层数量

    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256 # LSTM 内存单元大小（仅在 use_lstm=True 时使用）

    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer); works only when use_lstm is False
    D2RL = False # 是否使用 D2RL 结构（将最后一个卷积层的结果连接到每个隐藏层），仅在 use_lstm=False 时有效
    
    ### Training Params ###

    num_workers = 30 if not LOCAL_TESTING else 2 # 训练过程中使用的工作者数量

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0] # 所有用于实验的随机种子列表，用于结果复现

    # Placeholder for random for current trial
    seed = None # 当前试验的随机种子占位符

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1 # 主驱动程序应使用的 GPU 数量

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 12000 if not LOCAL_TESTING else 800 # 模拟的环境时间步数（跨所有环境），用于一次梯度更新。平均分配到各环境中

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 2000 if not LOCAL_TESTING else 800 # 进行梯度步骤前，将每个批次划分的小批量大小

    # Rollout length
    rollout_fragment_length = 400 # 回合片段长度

    # Whether all PPO agents should share the same policy network
    shared_policy = True # 所有 PPO 代理是否共享同一策略网络

    # Number of training iterations to run
    num_training_iters = 420 if not LOCAL_TESTING else 2 # 要运行的训练迭代次数

    # Stepsize of SGD.
    lr = 5e-5 # SGD 的步长

    # Learning rate schedule.
    lr_schedule = None # 学习率调度

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1 # 如果指定，按此值剪裁梯度的全局范数

    # Discount factor 
    gamma = 0.99 # 折扣因子

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98 # GAE 的指数衰减因子（蒙特卡洛样本的权重）

    # Whether the value function shares layers with the policy model
    vf_share_layers = True # 值函数是否与策略模型共享层
 
    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4   # 值网络在整体损失中的权重

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps # 熵奖励系数，将在 _start 和 _end 之间线性衰减，跨 _horizon 步数
    entropy_coeff_start = 0.2 
    entropy_coeff_end = 0.1
    entropy_coeff_horizon = 3e5

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2 # KL 散度的初始系数

    # PPO clipping factor
    clip_param = 0.05 # PPO 剪辑因子

    # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1 # 每个外部循环中的 SGD 迭代次数（每个训练批次执行的 epoch 数）

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 25 # 在保存模型检查点之前运行的训练迭代次数

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 1 # 每次评估之间运行的训练迭代次数

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400 # 评估回合的时间步数

    # Number of games to simulation each evaluation
    evaluation_num_games = 1   # 每次评估要模拟的游戏数量

    # Whether to display rollouts in evaluation
    evaluation_display = False # 是否在评估中显示回放

    # Where to log the ray dashboard stats # 用于记录 ray 仪表板统计数据的目录
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    # Where to store model checkpoints and training stats # 存储模型检查点和训练统计数据的目录
    results_dir = DEFAULT_RESULTS_DIR

    # Whether tensorflow should execute eagerly or not # Tensorflow 是否应该急切执行
    eager = False

    # Whether to log training progress and debugging info # 是否记录训练进度和调试信息
    verbose = True

    # -------- 似乎，上面的都是神经网络超参数，不太用理会 --------
    # -------- 下面的是要理解怎么调用 BC model 和 env 的参数，感觉需要理解参数的物理意义的 --------

    ### ---- BC Params (下面两个参数，都被包在了 bc_params 里) ---- ### 
    bc_model_dir = os.path.join(BC_SAVE_DIR, "default") # path to pickled policy model for behavior cloning (行为克隆策略模型的路径)    
    bc_stochastic = True # # Whether bc agents should return action logit argmax or sample (行为克隆代理是否返回动作对数几率 argmax 或样本)

    ### ---- Environment Params ---- ###
    
    layout_name = "cramped_room" # Which overcooked level to use (使用的 overcooked 关卡)

    # all_layout_names = '_'.join(layout_names) # 所有关卡名称的字符串表示

    # Name of directory to store training results in (stored in ~/ray_results/<experiment_name>) # 存储训练结果的目录名称（存储在 ~/ray_results/<experiment_name> 中）

    params_str = str(use_phi) + "_nw=%d_vf=%f_es=%f_en=%f_kl=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff,
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", layout_name, params_str)

    # Rewards the agent will receive for intermediate actions (代理将为中间动作接收的奖励)
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    # whether to start cooking automatically when pot has 3 items in it (当锅里有 3 个物品时是否自动开始烹饪)
    old_dynamics = False

    # Max episode length
    horizon = 400

    # Constant by which shaped rewards are multiplied by when calculating total reward (在计算总奖励时，形状奖励乘以的常数)
    reward_shaping_factor = 1.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps (奖励形状因子的线性退火，以使其在指定步数后达到零)
    reward_shaping_horizon = float("inf")

    # bc_factor represents that ppo agent gets paired with a bc agent for any episode (bc_factor 表示 ppo 代理是否在任何回合中与 bc 代理配对)
    # schedule for bc_factor is represented by a list of points (t_i, v_i) where v_i represents the (bc_factor 的计划由一系列点 (t_i, v_i) 表示，其中 v_i 表示时间步 t_i 处的 bc_factor 值)
    # value of bc_factor at timestep t_i. Values are linearly interpolated between points (值在点之间线性插值)
    # The default listed below represents bc_factor=0 for all timesteps (下面列出的默认值表示 bc_factor=0，适用于所有时间步)
    bc_schedule = OvercookedMultiAgent.self_play_bc_schedule # OvercookedMultiAgent 在 human_aware_rl.rllib.rllib 这个位置
    

    # To be passed into rl-lib model/custom_options config (传递给 rl-lib model/custom_options 配置)
    model_params = {
        "use_lstm": use_lstm,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "CELL_SIZE": CELL_SIZE,
        "D2RL": D2RL,
    }

    # to be passed into the rllib.PPOTrainer class (传递给 rllib.PPOTrainer 类)
    training_params = {
        "num_workers": num_workers,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "num_sgd_iter": num_sgd_iter,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "grad_clip": grad_clip,
        "gamma": gamma,
        "lambda": lmbda,
        "vf_share_layers": vf_share_layers,
        "vf_loss_coeff": vf_loss_coeff,
        "kl_coeff": kl_coeff,
        "clip_param": clip_param,
        "num_gpus": num_gpus,
        "seed": seed,
        "evaluation_interval": evaluation_interval,
        "entropy_coeff_schedule": [
            (0, entropy_coeff_start),
            (entropy_coeff_horizon, entropy_coeff_end),
        ],
        "eager_tracing": eager,
        "log_level": "WARN" if verbose else "ERROR",
    }

    # To be passed into AgentEvaluator constructor and _evaluate function (传递给 AgentEvaluator 构造函数和 _evaluate 函数)
    evaluation_params = {
        "ep_length": evaluation_ep_length,
        "num_games": evaluation_num_games,
        "display": evaluation_display,
    }

    environment_params = {
        # To be passed into OvercookedGridWorld constructor (传递给 OvercookedGridWorld 构造函数)
        "mdp_params": {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params,
            # old_dynamics == True makes cooking starts automatically without INTERACT (old_dynamics == True 表示烹饪在没有交互的情况下自动开始)
            # allows only 3-item recipes (仅允许 3 个物品的食谱)
            "old_dynamics": old_dynamics,
        },
        # To be passed into OvercookedEnv constructor (传递给 OvercookedEnv 构造函数)
        "env_params": {"horizon": horizon},
        # To be passed into OvercookedMultiAgent constructor (传递给 OvercookedMultiAgent 构造函数)
        "multi_agent_params": {
            "reward_shaping_factor": reward_shaping_factor,
            "reward_shaping_horizon": reward_shaping_horizon,
            "use_phi": use_phi,
            "bc_schedule": bc_schedule,
        },
    }

    bc_params = {
        "bc_policy_cls": BehaviorCloningPolicy,
        "bc_config": {
            "model_dir": bc_model_dir,
            "stochastic": bc_stochastic,
            "eager": eager,
        },
    }

    ray_params = {
        "custom_model_id": "MyPPOModel",
        "custom_model_cls": RllibLSTMPPOModel
        if model_params["use_lstm"]
        else RllibPPOModel,
        "temp_dir": temp_dir,
        "env_creator": _env_creator,
    }

    params = {
        "model_params": model_params,
        "training_params": training_params,
        "environment_params": environment_params,
        "bc_params": bc_params,
        "shared_policy": shared_policy,
        "num_training_iters": num_training_iters,
        "evaluation_params": evaluation_params,
        "experiment_name": experiment_name,
        "save_every": save_freq,
        "seeds": seeds,
        "results_dir": results_dir,
        "ray_params": ray_params,
        "resume_checkpoint_path": resume_checkpoint_path,
        "verbose": verbose,
    }


def run(params): # 定义 run 函数，参数 params 包含训练过程的所有配置。 而 run(params) 函数实现训练过程的具体逻辑。 
    run_name = params["experiment_name"] # 从参数中获取实验名称，并赋值给变量 run_name
    if params["verbose"]: # 如果参数 verbose 为 True，表示需要详细日志记录。
        import wandb # 导入 wandb 库，用于实验追踪和日志记录。

        wandb.init(project=WANDB_PROJECT, sync_tensorboard=True) # 初始化 wandb，设置项目名称为 WANDB_PROJECT 并同步 tensorboard
        wandb.run.name = run_name # 设置 wandb 运行名称为实验名称。
    
    # Retrieve the tune.Trainable object that is used for the experiment ??? tune.Trainable 什么鬼 ???
    # -------- 这里才是配置trainer的关键，下面循环里只是调用了一下这里配置好的trainer --------
    trainer = gen_trainer_from_params(params) # 调用 gen_trainer_from_params 函数 (human_aware_rl.rllib.rllib 所处脚本的位置)，根据参数生成一个训练器对象 trainer 
    # Object to store training results in
    result = {} # 初始化一个空字典 result，用于存储训练结果
    
    # -------- Training loop --------
    for i in range(params["num_training_iters"]):   # 根据参数中的训练迭代次数进行循环。
        if params["verbose"]:                       # 如果参数 verbose 为 True，打印训练迭代信息。
            print("Starting training iteration", i) # 打印当前迭代次数。
        
        # -------- 进行一次训练，并将结果存储在 result 变量中 --------
        result = trainer.train()
        # 包含下面三个步骤
        # data = trainer.worker.sample()
        # trainer.optimizer.update(data)
        # trainer.worker.sync_weight()     # 分享给分布式训练的其他同僚

        if i % params["save_every"] == 0: # 每隔一定次数的迭代保存一次模型。
            save_path = save_trainer(trainer, params) # 调用 save_trainer 函数，保存当前训练器的状态，并将保存路径存储在 save_path 变量中。

            if params["verbose"]: # 如果参数 verbose 为 True，打印保存路径。
                print("saved trainer at", save_path) # 打印保存路径。

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params) # 在训练结束时，调用 save_trainer 函数，保存最终的训练器状态，并将保存路径存储在 save_path 变量中。

    if params["verbose"]: # 如果参数 verbose 为 True，打印保存路径并结束 wandb 会话。
        print("saved trainer at", save_path) # 打印保存路径。
        # quiet = True so wandb doesn't log to console
        wandb.finish(quiet=True) # 结束 wandb 会话，设置 quiet=True 以便 wandb 不再向控制台记录日志。

    return result # 返回最后一次训练的结果。


# 当脚本运行时，sacred 会自动调用这个被 @ex.automain 装饰的函数，并将配置参数传递给它。
@ex.automain # 主函数使用 @ex.automain 装饰器，表示这是程序的入口点，调用 run(params) 开始训练
def main(params): # my_config 函数定义了实验的配置参数 params。 参数 params 包含实验的所有配置
    # List of each random seed to run，获取要运行的随机种子列表
    seeds = params["seeds"]
    del params["seeds"] # 从参数中删除 `seeds` 键，因为接下来不再需要这个键

    # this is required if we want to pass schedules in as command-line args, and we need to pass the string as a list of tuples 
    # 如果我们想从命令行参数传递计划，这里需要将字符串转 (string) 换为元组列表 (a list of tuples)
    bc_schedule = params["environment_params"]["multi_agent_params"]["bc_schedule"] # 获取多代理参数中的 bc_schedule 配置
    if not isinstance(bc_schedule[0], list):        # 检查 bc_schedule 的第一个元素是否为列表。如果不是列表，则进行转换。
        tuples_lst = []                             # 初始化一个空列表，用于存储转换后的元组。
        for i in range(0, len(bc_schedule), 2):     # 以步长为2遍历 bc_schedule 列表，用于提取成对的元素。
            x = int(bc_schedule[i].strip("("))      # 去除字符串的左括号并转换为整数。
            y = int(bc_schedule[i + 1].strip(")"))  # 去除字符串的右括号并转换为整数。
            tuples_lst.append((x, y))               # 将转换后的整数对作为元组添加到列表中
        params["environment_params"]["multi_agent_params"]["bc_schedule"] = tuples_lst # 将转换后的元组列表赋值回 bc_schedule 配置。

    # List to store results dicts (to be passed to sacred slack observer) # 用于存储结果字典的列表（传递给 sacred slack observer）
    results = [] # 初始化一个空列表，用于存储每次训练的结果

    # Train an agent to completion for each random seed specified # 为指定的每个随机种子训练一个代理到完成
    for seed in seeds:                              # 遍历每个随机种子，逐个进行训练
        # Override the seed                         # 覆盖种子
        params["training_params"]["seed"] = seed    # 将当前随机种子赋值给训练参数中的 seed 键

        # -------- 调用 run 函数执行训练，并获取结果 --------
        result = run(params)
        results.append(result) # 将训练结果添加到结果列表中。

    # Return value gets sent to our slack observer for notification # 返回的值将发送到我们的 slack observer 进行通知
    average_sparse_reward = np.mean([res["custom_metrics"]["sparse_reward_mean"] for res in results]) # 计算所有结果中稀疏奖励的平均值
    average_episode_reward = np.mean([res["episode_reward_mean"] for res in results]) # 计算所有结果中每集奖励的平均值

    return {
        "average_sparse_reward": average_sparse_reward,
        "average_total_reward": average_episode_reward,
    } # 返回一个包含平均稀疏奖励和平均总奖励的字典，供 sacred 的 slack observer 使用，用于通知实验结果。
