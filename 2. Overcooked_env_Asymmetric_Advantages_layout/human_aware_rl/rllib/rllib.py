import copy
import logging
import os
import random
import tempfile
from datetime import datetime

import dill
import gym
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.tune.result import DEFAULT_RESULTS_DIR

from human_aware_rl.rllib.utils import (
    get_base_ae,
    get_required_arguments,
    iterable_equal,
    softmax,
)
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedGridworld,
)

action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
obs_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


# RlLibAgent 类通过将 RLlib 策略对象封装，提供了一个兼容 Overcooked 环境的智能体实现。它包括初始化、重置、计算动作概率和执行动作的方法，使得这个智能体可以与 Overcooked 环境进行交互。
class RlLibAgent(Agent): # Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent， RlLibAgent 类将训练好的 RLlib 策略对象包装成一个兼容 Overcooked 环境的智能体。 

    def __init__(self, policy, agent_index, featurize_fn): # 初始化方法，接受策略对象、智能体索引和特征提取函数
        self.policy = policy
        self.agent_index = agent_index
        self.featurize = featurize_fn

    def reset(self): #  重置方法，获取初始的 RNN 状态并添加批次维度。 如果策略模型或策略本身有初始状态的方法，就使用它们来初始化 RNN 状态
        # Get initial rnn states and add batch dimension to each
        if hasattr(self.policy.model, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.model.get_initial_state()
            ]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [
                np.expand_dims(state, axis=0)
                for state in self.policy.get_initial_state()
            ]
        else:
            self.rnn_state = []

    def action_probabilities(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - Normalized action probabilities determined by self.policy
        """
        # Preprocess the environment state，预处理环境状态，将其特征化
        obs = self.featurize(state, debug=False)
        my_obs = obs[self.agent_index]

        # Compute non-normalized log probabilities from the underlying model，从策略模型计算非标准化的对数概率
        logits = self.policy.compute_actions(
            np.array([my_obs]), self.rnn_state
        )[2]["action_dist_inputs"]

        # Softmax in numpy to convert logits to normalized probabilities，使用 Softmax 函数将对数概率转换为标准化的概率，并返回
        return softmax(logits)

    def action(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """
        # Preprocess the environment state，预处理环境状态，将其特征化
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]

        # Use Rllib.Policy class to compute action argmax and action probabilities，使用 Rllib.Policy 类计算动作的 argmax 和动作概率
        # The first value is action_idx, which we will recompute below so the results are stochastic
        _, rnn_state, info = self.policy.compute_actions(
            np.array([my_obs]), self.rnn_state
        )

        # Softmax in numpy to convert logits to normalized probabilities，使用 Softmax 函数将对数概率转换为标准化的概率
        logits = info["action_dist_inputs"]
        action_probabilities = softmax(logits)

        # The original design is stochastic across different games,
        # Though if we are reloading from a checkpoint it would inherit the seed at that point, producing deterministic results
        # 原始设计在不同的游戏中是随机的，使用随机选择方法选择动作索引，然后转换为实际动作
        [action_idx] = random.choices(
            list(range(Action.NUM_ACTIONS)), action_probabilities[0]
        )
        agent_action = Action.INDEX_TO_ACTION[action_idx]

        agent_action_info = {"action_probs": action_probabilities}
        self.rnn_state = rnn_state # 创建动作信息字典，存储动作概率，并更新 RNN 状态

        return agent_action, agent_action_info  # 返回动作和动作信息字典


class OvercookedMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    # List of all agent types currently supported
    supported_agents = ["ppo", "bc"]

    # Default bc_schedule, includes no bc agent at any time
    bc_schedule = self_play_bc_schedule = [(0, 0), (float("inf"), 0)]

    # Default environment params used for creation
    DEFAULT_CONFIG = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params": {
            "layout_name": "cramped_room",
            "rew_shaping_params": {},
        },
        # To be passed into OvercookedEnv constructor
        "env_params": {"horizon": 400},
        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params": {
            "reward_shaping_factor": 0.0,
            "reward_shaping_horizon": 0,
            "bc_schedule": self_play_bc_schedule,
            "use_phi": True,
        },
    }

    def __init__(
        self,
        base_env,
        reward_shaping_factor=0.0,
        reward_shaping_horizon=0,
        bc_schedule=None,
        use_phi=True,
    ):
        """
        base_env: OvercookedEnv
        reward_shaping_factor (float): Coefficient multiplied by dense reward before adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor reaches zero through linear annealing
        bc_schedule (list[tuple]): List of (t_i, v_i) pairs where v_i represents the value of bc_factor at timestep t_i
            with linear interpolation in between the t_i
        use_phi (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        """
        if bc_schedule:
            self.bc_schedule = bc_schedule
        self._validate_schedule(self.bc_schedule)
        self.base_env = base_env
        # since we are not passing featurize_fn in as an argument, we create it here and check its validity
        self.featurize_fn_map = {
            "ppo": lambda state: self.base_env.lossless_state_encoding_mdp(
                state
            ),
            "bc": lambda state: self.base_env.featurize_state_mdp(state),
        }
        self._validate_featurize_fns(self.featurize_fn_map)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self.use_phi = use_phi
        self.anneal_bc_factor(0)
        self._agent_ids = set(self.reset().keys())
        # fixes deprecation warnings
        self._spaces_in_preferred_format = True

    def _validate_featurize_fns(self, mapping):
        assert "ppo" in mapping, "At least one ppo agent must be specified"
        for k, v in mapping.items():
            assert (
                k in self.supported_agents
            ), "Unsuported agent type in featurize mapping {0}".format(k)
            assert callable(v), "Featurize_fn values must be functions"
            assert (
                len(get_required_arguments(v)) == 1
            ), "Featurize_fn value must accept exactly one argument"

    def _validate_schedule(self, schedule):
        timesteps = [p[0] for p in schedule]
        values = [p[1] for p in schedule]

        assert (
            len(schedule) >= 2
        ), "Need at least 2 points to linearly interpolate schedule"
        assert schedule[0][0] == 0, "Schedule must start at timestep 0"
        assert all(
            [t >= 0 for t in timesteps]
        ), "All timesteps in schedule must be non-negative"
        assert all(
            [v >= 0 and v <= 1 for v in values]
        ), "All values in schedule must be between 0 and 1"
        assert (
            sorted(timesteps) == timesteps
        ), "Timesteps must be in increasing order in schedule"

        # To ensure we flatline after passing last timestep
        if schedule[-1][0] < float("inf"):
            schedule.append((float("inf"), schedule[-1][1]))

    def _setup_action_space(self, agents):
        action_sp = {}
        for agent in agents:
            action_sp[agent] = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.action_space = gym.spaces.Dict(action_sp)
        self.shared_action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

    def _setup_observation_space(self, agents):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        # ppo observation
        featurize_fn_ppo = (
            lambda state: self.base_env.lossless_state_encoding_mdp(state)
        )
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape

        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(
            state
        )
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(
            np.float32(low), np.float32(high), dtype=np.float32
        )
        # hardcode mapping between action space and agent
        ob_space = {}
        for agent in agents:
            if agent.startswith("ppo"):
                ob_space[agent] = self.ppo_observation_space
            else:
                ob_space[agent] = self.bc_observation_space
        self.observation_space = gym.spaces.Dict(ob_space)

    def _get_featurize_fn(self, agent_id):
        if agent_id.startswith("ppo"):
            return lambda state: self.base_env.lossless_state_encoding_mdp(
                state
            )
        if agent_id.startswith("bc"):
            return lambda state: self.base_env.featurize_state_mdp(state)
        raise ValueError("Unsupported agent type {0}".format(agent_id))

    def _get_obs(self, state):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]
        return ob_p0.astype(np.float32), ob_p1.astype(np.float32)

    def _populate_agents(self):
        # Always include at least one ppo agent (i.e. bc_sp not supported for simplicity)
        agents = ["ppo"]

        # Coin flip to determine whether other agent should be ppo or bc
        other_agent = "bc" if np.random.uniform() < self.bc_factor else "ppo"
        agents.append(other_agent)

        # Randomize starting indices
        np.random.shuffle(agents)

        # Ensure agent names are unique
        agents[0] = agents[0] + "_0"
        agents[1] = agents[1] + "_1"

        # logically the action_space and the observation_space should be set along with the generated agents
        # the agents are also randomized in each iteration if bc agents are allowed, which requires reestablishing the action & observation space
        self._setup_action_space(agents)
        self._setup_observation_space(agents)
        return agents

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        action = [
            action_dict[self.curr_agents[0]],
            action_dict[self.curr_agents[1]],
        ]

        assert all(
            self.action_space[agent].contains(action_dict[agent])
            for agent in action_dict
        ), "%r (%s) invalid" % (action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment

        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=True
            )
            potential = info["phi_s_prime"] - info["phi_s"]
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(
                joint_action, display_phi=False
            )
            dense_reward = info["shaped_r_by_agent"]

        ob_p0, ob_p1 = self._get_obs(next_state)

        shaped_reward_p0 = (
            sparse_reward + self.reward_shaping_factor * dense_reward[0]
        )
        shaped_reward_p1 = (
            sparse_reward + self.reward_shaping_factor * dense_reward[1]
        )

        obs = {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}
        rewards = {
            self.curr_agents[0]: shaped_reward_p0,
            self.curr_agents[1]: shaped_reward_p1,
        }
        dones = {
            self.curr_agents[0]: done,
            self.curr_agents[1]: done,
            "__all__": done,
        }
        infos = {self.curr_agents[0]: info, self.curr_agents[1]: info}
        return obs, rewards, dones, infos

    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        ob_p0, ob_p1 = self._get_obs(self.base_env.state)
        return {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(
            self._initial_reward_shaping_factor,
            timesteps,
            self.reward_shaping_horizon,
        )
        self.set_reward_shaping_factor(new_factor)

    def anneal_bc_factor(self, timesteps):
        """
        Set the current bc factor such that we anneal linearly until self.bc_factor_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        p_0 = self.bc_schedule[0]
        p_1 = self.bc_schedule[1]
        i = 2
        while timesteps > p_1[0] and i < len(self.bc_schedule):
            p_0 = p_1
            p_1 = self.bc_schedule[i]
            i += 1
        start_t, start_v = p_0
        end_t, end_v = p_1
        new_factor = self._anneal(start_v, timesteps, end_t, end_v, start_t)
        self.set_bc_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def set_bc_factor(self, factor):
        self.bc_factor = factor

    def seed(self, seed):
        """
        set global random seed to make environment deterministic
        """
        # Our environment is already deterministic
        pass

    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedMultiAgent constuctor

        Returns:
            OvercookedMultiAgent instance specified by env_config params
        """
        assert (
            env_config
            and "env_params" in env_config
            and "multi_agent_params" in env_config
        )
        assert (
            "mdp_params" in env_config
            or "mdp_params_schedule_fn" in env_config
        ), "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "rew_shaping_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(
            mdp_params, env_params, outer_shape, mdp_params_schedule_fn
        )
        base_env = base_ae.env

        return cls(base_env, **multi_agent_params)


##################
# Training Utils #
##################


class TrainingCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        pass

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data

        sparse_reward (int) - total reward from deliveries agent earned this episode
        shaped_reward (int) - total reward shaping reward the agent earned this episode
        """
        # Get rllib.OvercookedMultiAgentEnv refernce from rllib wraper
        env = base_env.get_sub_environments()[0]
        # Both agents share the same info so it doesn't matter whose we use, just use 0th agent's
        info_dict = episode.last_info_for(env.curr_agents[0])

        ep_info = info_dict["episode"]
        game_stats = ep_info["ep_game_stats"]

        # List of episode stats we'd like to collect by agent
        stats_to_collect = EVENT_TYPES

        # Parse info dicts generated by OvercookedEnv
        tot_sparse_reward = ep_info["ep_sparse_r"]
        tot_shaped_reward = ep_info["ep_shaped_r"]

        # Store metrics where they will be visible to rllib for tensorboard logging
        episode.custom_metrics["sparse_reward"] = tot_sparse_reward
        episode.custom_metrics["shaped_reward"] = tot_shaped_reward

        # Store per-agent game stats to rllib info dicts
        for stat in stats_to_collect:
            stats = game_stats[stat]
            episode.custom_metrics[stat + "_agent_0"] = len(stats[0])
            episode.custom_metrics[stat + "_agent_1"] = len(stats[1])

    def on_sample_end(self, worker, samples, **kwargs):
        pass

    # Executes at the end of a call to Trainer.train, we'll update environment params (like annealing shaped rewards)
    def on_train_result(self, trainer, result, **kwargs):
        # Anneal the reward shaping coefficient based on environment paremeters and current timestep
        timestep = result["timesteps_total"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_reward_shaping_factor(timestep)
            )
        )

        # Anneal the bc factor based on environment paremeters and current timestep
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_bc_factor(timestep)
            )
        )

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        pass


# get_rllib_eval_function 函数的作用是生成一个符合 RLlib 自定义评估函数签名的 _evaluate 函数。这个 _evaluate 函数在评估过程中随机选择智能体的策略，并计算相应的评估结果。通过将必要的评估参数封装在本地作用域中，简化了评估函数的定义和调用。
def get_rllib_eval_function(
    eval_params,                # (dict)：包含评估所需的参数，如 num_games（游戏数量），display（是否显示），ep_length（每局长度）
    eval_mdp_params,            # (dict)：用于创建基础的 OvercookedMDP，MDP表示马尔科夫决策过程
    env_params,                 # (dict)：用于创建基础的 OvercookedEnv，这是一个强化学习环境
    outer_shape,                # (list)：评估布局的外部形状，包含两个元素的列表
    agent_0_policy_str="ppo",   #  (str)：第一个智能体使用的策略，默认为 "ppo"
    agent_1_policy_str="ppo",   #  (str)：第二个智能体使用的策略，默认为 "ppo"
    verbose=False,              # (bool)：是否打印详细信息，默认为 False
):
    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a
    function with rllib custom_evaluation_function compatible signature

    eval_params (dict): Contains 'num_games' (int), 'display' (bool), and 'ep_length' (int)
    mdp_params (dict): Used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    agent_1_policy_str (str): Key associated with the rllib policy object used to select actions (must be either 'ppo' or 'bc')
    Note: Agent policies are shuffled each time, so agent_0_policy_str and agent_1_policy_str are symmetric
    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers): # 这是嵌套函数 _evaluate，它是实际执行评估的函数。
        if verbose: # 如果 verbose 为 True，打印评估开始的信息。
            print("Computing rollout of current trained policy")

        # Randomize starting indices 随机打乱策略列表，确保每次评估时智能体使用的策略是随机的。agent_0_policy 和 agent_1_policy 分别被设置为打乱后的策略
        policies = [agent_0_policy_str, agent_1_policy_str]
        np.random.shuffle(policies)
        agent_0_policy, agent_1_policy = policies

        # Get the corresponding rllib policy objects for each policy string name 获取对应策略字符串名称的 RLlib 策略对象
        agent_0_policy = trainer.get_policy(agent_0_policy)
        agent_1_policy = trainer.get_policy(agent_1_policy)

        # 如果策略中包含 "bc"（行为克隆），则初始化相应的特征提取函数 bc_featurize_fn
        # 如果 policies[0] 是 "bc"，则将 agent_0_feat_fn 设置为 bc_featurize_fn
        # 如果 policies[1] 是 "bc"，则将 agent_1_feat_fn 设置为 bc_featurize_fn
        agent_0_feat_fn = agent_1_feat_fn = None
        if "bc" in policies:
            base_ae = get_base_ae(eval_mdp_params, env_params)
            base_env = base_ae.env
            bc_featurize_fn = lambda state: base_env.featurize_state_mdp(state)
            if policies[0] == "bc":
                agent_0_feat_fn = bc_featurize_fn
            if policies[1] == "bc":
                agent_1_feat_fn = bc_featurize_fn

        # Compute the evauation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
        # 调用 evaluate 函数进行评估。evaluate 函数计算评估结果，参数包括评估参数、MDP 参数、外部形状、智能体策略和特征提取函数。
        results = evaluate(
            eval_params,
            eval_mdp_params,
            outer_shape,
            agent_0_policy,
            agent_1_policy,
            agent_0_feat_fn,
            agent_1_feat_fn,
            verbose=verbose,
        )

        # Log any metrics we care about for rllib tensorboard visualization
        # 创建一个字典 metrics 来存储评估结果。计算每局的平均稀疏奖励，并将其存储在 metrics 中，然后返回 metrics
        metrics = {}
        metrics["average_sparse_reward"] = np.mean(results["ep_returns"])
        return metrics

    return _evaluate # 返回嵌套的 _evaluate 函数，使其成为一个符合 RLlib 自定义评估函数签名的函数。


def evaluate(
    eval_params,
    mdp_params,
    outer_shape,
    agent_0_policy,
    agent_1_policy,
    agent_0_featurize_fn=None,
    agent_1_featurize_fn=None,
    verbose=False,
):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to lossless_state_encoding if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to lossless_state_encoding if 'None'
    """
    if verbose:
        print("eval mdp params", mdp_params)
    evaluator = get_base_ae(
        mdp_params,
        {"horizon": eval_params["ep_length"], "num_mdp": 1},
        outer_shape,
    )

    # Override pre-processing functions with defaults if necessary
    agent_0_featurize_fn = (
        agent_0_featurize_fn
        if agent_0_featurize_fn
        else evaluator.env.lossless_state_encoding_mdp
    )
    agent_1_featurize_fn = (
        agent_1_featurize_fn
        if agent_1_featurize_fn
        else evaluator.env.lossless_state_encoding_mdp
    )

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = RlLibAgent(agent_0_policy, agent_index=0, featurize_fn=agent_0_featurize_fn)
    agent1 = RlLibAgent(agent_1_policy, agent_index=1, featurize_fn=agent_1_featurize_fn)

    # Compute rollouts
    if "store_dir" not in eval_params:
        eval_params["store_dir"] = None
    if "display_phi" not in eval_params:
        eval_params["display_phi"] = False
    results = evaluator.evaluate_agent_pair(
        AgentPair(agent0, agent1),
        num_games=eval_params["num_games"],
        display=eval_params["display"],
        dir=eval_params["store_dir"],
        display_phi=eval_params["display_phi"],
        info=verbose,
    )

    return results


###########################
# rllib.Trainer functions #
###########################

# -------- 训练过程实际上是由 Ray RLlib 框架管理的，具体体现在 gen_trainer_from_params 函数和 PPOTrainer 类的使用中 --------
# 训练过程概述: 在 Ray RLlib 框架中，训练过程通过 PPOTrainer 类（或其他类似的 Trainer 类）实现。PPOTrainer 类封装了策略（policy）和环境（environment）的交互过程，包括状态（state）和动作（action）的更新，以及策略参数的优化。
# 以下是训练过程的关键部分：
# 创建环境和策略：在 gen_trainer_from_params 函数中，通过 register_env 和 ModelCatalog.register_custom_model 注册环境和自定义模型。这些配置是训练的基础。
# 配置训练参数：gen_trainer_from_params 函数解析并组织训练参数，包括多智能体配置、训练配置、环境配置等。
# 创建 PPO 训练器：通过调用 PPOTrainer 类实例化一个训练器对象，这个对象包含了环境、策略、训练配置、回调函数等。
# 训练循环：在 run 函数 (ppo_rllib_client.py脚本里) 中，使用 trainer.train() 方法进行迭代训练。每次调用 trainer.train()，PPO 训练器会在环境中采样、执行动作、收集数据、更新策略参数等。
 
# -------- 在 ppo_rllib_client.py 脚本中调用了这个关键函数，使用的是如下的命令: trainer = gen_trainer_from_params(params) --------
# 这个函数的作用是根据给定的参数生成并配置一个用于训练的 PPOTrainer 对象
def gen_trainer_from_params(params): # 定义 gen_trainer_from_params 函数，参数 params 包含生成训练器所需的所有配置
    # All ray environment set-up
    if not ray.is_initialized(): # 检查 ray 是否已经初始化，如果没有，则进行初始化。
        init_params = {
            "ignore_reinit_error": True,                                              # 忽略重新初始化错误
            "include_dashboard": False,                                               # 不包含仪表盘
            "_temp_dir": params["ray_params"]["temp_dir"],                            # 设置临时目录
            "log_to_driver": params["verbose"],                                       # 根据 verbose 参数决定是否记录日志到驱动程序
            "logging_level": logging.INFO if params["verbose"] else logging.CRITICAL, # 设置日志记录级别
        }
        ray.init(**init_params) # 初始化 ray
    # -- 环境和策略的注册 --
    register_env("overcooked_multi_agent", params["ray_params"]["env_creator"]) # 注册自定义环境，名称为 "overcooked_multi_agent"。
    ModelCatalog.register_custom_model(                                         # 注册自定义模型
        params["ray_params"]["custom_model_id"],
        params["ray_params"]["custom_model_cls"],
    )
    # Parse params
    model_params = params["model_params"]                                   # 获取模型参数
    training_params = params["training_params"]                             # 获取训练参数
    environment_params = params["environment_params"]                       # 获取环境参数
    evaluation_params = params["evaluation_params"]                         # 获取评估参数
    bc_params = params["bc_params"]                                         # 获取行为克隆参数
    multi_agent_params = params["environment_params"]["multi_agent_params"] # 获取多智能体参数

    env = OvercookedMultiAgent.from_config(environment_params) # 根据环境参数创建 OvercookedMultiAgent 实例

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy(policy_type="ppo"): # 定义生成策略的内部函数
        # supported policy types thus far
        assert policy_type in ["ppo", "bc"] # 确认策略类型是 "ppo" 或 "bc"

        if policy_type == "ppo": # 如果策略类型是 "ppo"
            config = { # 设置 PPO 策略的配置
                "model": {
                    "custom_model_config": model_params,
                    "custom_model": "MyPPOModel",
                }
            }
            return ( # 返回 PPO 策略元组
                None,
                env.ppo_observation_space,
                env.shared_action_space,
                config,
            )
        elif policy_type == "bc": # 如果策略类型是 "bc"
            bc_cls = bc_params["bc_policy_cls"] # 获取行为克隆策略类
            bc_config = bc_params["bc_config"]  # 获取行为克隆配置
            return ( # 返回行为克隆策略元组
                bc_cls,
                env.bc_observation_space,
                env.shared_action_space,
                bc_config,
            )

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = "{0}_{1}_{2}".format(params["experiment_name"], params["training_params"]["seed"], timestr) # 生成日志目录前缀

    def custom_logger_creator(config): # 定义自定义日志创建函数。
        """Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>"""
        results_dir = params["results_dir"] # 获取结果目录
        if not os.path.exists(results_dir): # 如果结果目录不存在，则创建
            try:                            # 尝试创建结果目录
                os.makedirs(results_dir)
            except Exception as e:          # 如果创建失败，打印错误信息并使用默认目录
                print(
                    "error creating custom logging dir. Falling back to default logdir {}".format(
                        DEFAULT_RESULTS_DIR
                    )
                )
                results_dir = DEFAULT_RESULTS_DIR
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=results_dir) # 创建临时日志目录
        logger = UnifiedLogger(config, logdir, loggers=None) # 创建统一 logger
        return logger # 返回 logger

    # Create rllib compatible multi-agent config based on params
    multi_agent_config = {} # 初始化多智能体配置字典
    all_policies = ["ppo"]  # 默认策略为 PPO

    # Whether both agents should be learned
    self_play = iterable_equal( # 检查是否为自对弈模式
        multi_agent_params["bc_schedule"],
        OvercookedMultiAgent.self_play_bc_schedule,
    )
    if not self_play: # 如果不是自对弈，添加 BC 策略
        all_policies.append("bc")

    multi_agent_config["policies"] = {policy: gen_policy(policy) for policy in all_policies} # 生成并设置所有策略

    def select_policy(agent_id, episode, worker, **kwargs): # 定义策略选择函数
        if agent_id.startswith("ppo"): # 如果代理 ID 以 "ppo" 开头，返回 "ppo" 策略
            return "ppo"
        if agent_id.startswith("bc"):  # 如果代理 ID 以 "bc" 开头，返回 "bc" 策略
            return "bc"

    multi_agent_config["policy_mapping_fn"] = select_policy # 设置策略映射函数
    multi_agent_config["policies_to_train"] = {"ppo"}       # 设置要训练的策略

    if "outer_shape" not in environment_params: # 如果环境参数中没有 "outer_shape"，设置为 None
        environment_params["outer_shape"] = None

    if "mdp_params" in environment_params: # 如果环境参数中有 "mdp_params"，设置评估 MDP 参数
        environment_params["eval_mdp_params"] = environment_params[
            "mdp_params"
        ]

    # -------- 创建 PPO 训练器，设置环境、配置、回调、自定义评估函数、日志创建函数等 --------
    # multi_agent_config：   多智能体配置：定义了多智能体环境中的策略和策略映射函数。包括了每个智能体使用的策略以及如何在不同的智能体之间进行策略分配。
    # TrainingCallbacks：    训练回调：用于在训练的不同阶段执行特定的操作，例如在每次训练迭代结束后记录一些指标或在特定条件下保存模型。
    # custom_eval_function： 自定义评估函数：允许用户定义自己的评估逻辑，而不是使用默认的评估方式。 这对需要特定评估指标或复杂评估逻辑的情况非常有用。
    # env_config：           环境配置：包含了初始化环境所需的所有参数。这些参数定义了环境的行为和特性。
    # eager_tracing：        Eager 模式：在 TensorFlow 中，Eager 模式是一种即时执行模式。设置为 False 表示不使用 Eager 模式，可能是因为 RLlib 默认使用静态图执行模式，这对于性能优化更有利。
    # training_params：      训练配置：包括了训练过程中的各项配置，如学习率、折扣因子、训练的迭代次数等。这些参数直接影响训练过程和结果。
    # logger_creator：       自定义日志创建函数：用于创建自定义的日志记录器。通过这种方式，可以将训练过程中的日志信息存储到指定的目录中，并以特定的格式进行记录。

    # -- 在 Ray RLlib 框架中，训练过程通过 PPOTrainer 类（或其他类似的 Trainer 类）实现 --
    # -- PPOTrainer 类封装了策略（policy）和环境（environment）的交互过程，包括状态（state）和动作（action）的更新，以及策略参数的优化 --
    trainer = PPOTrainer(
        env="overcooked_multi_agent",                           # 指定训练环境的名称，这里是 "overcooked_multi_agent"
        config={                                                # 这是一个包含训练配置的字典，具体包括以下子项
            "multiagent": multi_agent_config,                   # 多智能体配置，包含策略定义、策略映射函数等。 multi_agent_config 的具体内容已经在前面定义了
            "callbacks": TrainingCallbacks,                     # 训练回调，用于在训练过程中执行自定义操作。 TrainingCallbacks 是一个包含回调函数的类，用于在训练的不同阶段执行特定的操作, 例如在每次训练迭代结束后记录一些指标或在特定条件下保存模型。
            "custom_eval_function": get_rllib_eval_function(    # -------- 一个自定义评估函数，源代码在上面，用于在训练过程中评估策略的表现, 它接受多个参数： --------
                evaluation_params,                              # 评估参数
                environment_params["eval_mdp_params"],          # 评估 MDP 参数
                environment_params["env_params"],               # 环境参数
                environment_params["outer_shape"],              # 外部形状参数
                "ppo",                                          # 策略类型
                "ppo" if self_play else "bc",                   # 另一个策略类型，如果是自对弈模式则为 "ppo"，否则为 "bc"
                verbose=params["verbose"],
            ),
            "env_config": environment_params,                   # 环境配置，直接传入 environment_params，包含环境初始化所需的所有参数
            "eager_tracing": False,                             # 是否启用 Eager 模式。 Eager 模式是 TensorFlow 的一种即时执行模式，但这里设置为 False
            **training_params,                                  # 其他训练配置参数 (也是在params当中的一个部分)，这里使用 Python 的解包语法将 training_params 中的所有键值对添加到配置字典中
        },
        logger_creator=custom_logger_creator,                   # 自定义日志创建函数，用于创建自定义的日志记录器。custom_logger_creator 是一个函数，返回一个 logger 实例，用于记录训练过程中的信息
    )
    return trainer                                              # 返回创建的训练器


### Serialization ###


def save_trainer(trainer, params, path=None):
    """
    Saves a serialized trainer checkpoint at `path`. If none provided, the default path is
    ~/ray_results/<experiment_results_dir>/checkpoint_<i>
    Note that `params` should follow the same schema as the dict passed into `gen_trainer_from_params`
    """
    # Save trainer
    save_path = trainer.save(path)

    # Save params used to create trainer in /path/to/checkpoint_dir/config.pkl
    config = copy.deepcopy(params)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")

    # Note that we use dill (not pickle) here because it supports function serialization
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path


def load_trainer(save_path, true_num_workers=False):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint directory
    Additionally we decide if we want to use the same number of remote workers (see ray library Training APIs)
    as we store in the previous configuration, by default = False, we use only the local worker
    (see ray library API)
    """
    # Read in params used to create trainer
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    if not true_num_workers:
        # Override this param to lower overhead in trainer creation
        config["training_params"]["num_workers"] = 0

    if config["training_params"]["num_gpus"] == 1:
        # all other configs for the server can be kept for local testing
        config["training_params"]["num_gpus"] = 0

    if "trained_example" in save_path:
        # For the unit testing we update the result directory in order to avoid an error
        config[
            "results_dir"
        ] = "/Users/runner/work/human_aware_rl/human_aware_rl/human_aware_rl/ppo/results_temp"

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)
    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer


def get_agent_from_trainer(trainer, policy_id="ppo", agent_index=0):
    policy = trainer.get_policy(policy_id)
    dummy_env = trainer.env_creator(trainer.config["env_config"])
    featurize_fn = dummy_env.featurize_fn_map[policy_id]
    agent = RlLibAgent(policy, agent_index, featurize_fn=featurize_fn)
    return agent


def get_agent_pair_from_trainer(trainer, policy_id_0="ppo", policy_id_1="ppo"):
    agent0 = get_agent_from_trainer(trainer, policy_id=policy_id_0)
    agent1 = get_agent_from_trainer(trainer, policy_id=policy_id_1)
    return AgentPair(agent0, agent1)


def load_agent_pair(save_path, policy_id_0="ppo", policy_id_1="ppo"):
    """
    Returns an Overcooked AgentPair object that has as player 0 and player 1 policies with
    ID policy_id_0 and policy_id_1, respectively
    """
    trainer = load_trainer(save_path)
    return get_agent_pair_from_trainer(trainer, policy_id_0, policy_id_1)


def load_agent(save_path, policy_id="ppo", agent_index=0):
    """
    Returns an RllibAgent (compatible with the Overcooked Agent API) from the `save_path` to a previously
    serialized trainer object created with `save_trainer`

    The trainer can have multiple independent policies, so extract the one with ID `policy_id` to wrap in
    an RllibAgent

    Agent index indicates whether the agent is player zero or player one (or player n in the general case)
    as the featurization is not symmetric for both players
    """
    trainer = load_trainer(save_path)
    return get_agent_from_trainer(
        trainer, policy_id=policy_id, agent_index=agent_index
    )
