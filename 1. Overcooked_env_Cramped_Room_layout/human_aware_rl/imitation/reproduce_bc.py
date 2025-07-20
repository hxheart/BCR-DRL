import os

from human_aware_rl.imitation.behavior_cloning_tf2 import (
    get_bc_params,
    train_bc_model,
)
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_TRAIN,
)

# random 3 is counter_circuit
# random 0 is forced coordination
# the reason why we use these as the layouts name here is that in the cleaned pickled file of human trajectories, the df has layout named random3 and random0
# So in order to extract the right data from the df, we need to use these names
# however when loading layouts there are no random0/3
# The same parameter is used in both setting up the layout for training and loading the corresponding trajectories
# so without modifying the dataframes, I have to create new layouts
if __name__ == "__main__":
    for layout in [
        "cramped_room",                     # 1
        "asymmetric_advantages",            # 2
        "coordination_ring",                # 3
        "random0",  # forced coordination   # 4
        "random3",  # counter_circuit       # 5
    ]:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))       # 获取当前文件所在的目录路径，并将其赋值给current_file_dir变量
        bc_dir = os.path.join(current_file_dir, "bc_runs", "train", layout) # 使用os.path.join将当前文件目录路径current_file_dir与子目录"bc_runs/train/layout"拼接在一起，生成一个新的路径。其中layout个变量，是上面5个中的一个
        if os.path.isdir(bc_dir):   # 检查路径bc_dir是否是一个存在的目录。 os.path.isdir函数返回一个布尔值，表示该路径是否是一个目录
            continue                # 如果bc_dir目录存在，则跳过当前循环的剩余部分，继续下一个布局的处理
        
        params_to_override = {      # 创建一个字典params_to_override，用于存储要覆盖的参数。这个字典包含以下键值对：
            "layouts": [layout],    # 包含当前布局的列表
            "layout_name": layout,  # 当前布局的名称
            "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,   # 训练数据的路径  
            "epochs": 100,          # 训练的轮数
            "old_dynamics": True,   # 一个布尔值参数，设为True，表示使用旧的动态
        }
        bc_params = get_bc_params(**params_to_override) # 调用函数get_bc_params，传入参数params_to_override，并将返回的结果赋值给bc_params变量。**params_to_override表示将字典中的键值对作为命名参数传递给函数
        train_bc_model(bc_dir, bc_params, True)


# ---- 关于 current_file_dir, 举例讲一下就是 ----
# 假设脚本位于/home/user/project/script.py：
# __file__ = script.py
# os.path.abspath(__file__) = /home/user/project/script.py
# os.path.dirname(os.path.abspath(__file__)) = /home/user/project


# ---- 关于 get_bc_params 参数传递, 举例讲一下就是 ----
# 假设params_to_override的内容是：
# {
#     "layouts": [layout],
#     "layout_name": layout,
#     "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
#     "epochs": 100,
#     "old_dynamics": True
# }
# 传递给get_bc_params函数时，实际上等效于：
# get_bc_params(
#     layouts=[layout],
#     layout_name=layout,
#     data_path=CLEAN_2019_HUMAN_DATA_TRAIN,
#     epochs=100,
#     old_dynamics=True
# )