# ppo_rllib_client.py 负责训练代理，而 plot_example_experiments.py 则负责对训练结果进行分析和可视化。
# 两者配合使用，可以完成从训练到结果分析的整个流程。


import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from human_aware_rl.utils import *
from human_aware_rl.utils import set_style

envs = [ # 列出了所有感兴趣的环境（布局）
    "cramped_room",
    "forced_coordination",
    "counter_circuit_o_1",
    "coordination_ring",
    "asymmetric_advantages",
]


def get_list_experiments(path): # 该函数遍历指定目录（path），找到所有子目录，并将与环境名称匹配的子目录路径存储在一个字典中。
    result = {}
    subdirs = [
        name
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    for env in envs:
        result[env] = {
            "files": [path + "/" + x for x in subdirs if re.search(env, x)]
        }
    return result


def get_statistics(dict): # 该函数从每个实验的结果文件中提取最后一episode的平均稀疏奖励，计算每个环境的平均值和标准差，并将这些统计数据添加到字典中。
    for env in dict:
        rewards = [
            get_last_episode_rewards(file + "/result.json")[
                "sparse_reward_mean"
            ]
            for file in dict[env]["files"]
        ]
        dict[env]["rewards"] = rewards
        dict[env]["std"] = np.std(rewards)
        dict[env]["mean"] = np.mean(rewards)
    return dict


def plot_statistics(dict): # 该函数将统计数据绘制成条形图，展示每个环境的平均奖励和标准差。
    names = []
    stds = []
    means = []
    for env in dict:
        names.append(env)
        stds.append(dict[env]["std"])
        means.append(dict[env]["mean"])

    x_pos = np.arange(len(names))
    matplotlib.rc("xtick", labelsize=7)
    fig, ax = plt.subplots()
    ax.bar(
        x_pos,
        means,
        yerr=stds,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    ax.set_ylabel("Average reward per episode")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig("example_rewards.png")
    plt.show()


if __name__ == "__main__": # 主程序先获取实验列表，然后计算统计数据，最后绘制并展示结果。
    experiments = get_list_experiments("results")
    experiments_results = get_statistics(experiments)
    plot_statistics(experiments_results)
