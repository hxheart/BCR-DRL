import matplotlib.pyplot as plt
import numpy as np

fontsize_figure = 14  # 字体大小
# xlim_max = 600
# ylim_max = 150

def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# 数据加载函数（可以替换为你的实际数据加载逻辑）
def load_data(file_pattern, num_runs_begin, num_runs_end):
    data_all = []
    for index_run in range(num_runs_begin, num_runs_end):
        data_all.append(np.load(file_pattern.format(index_run)))
    return np.array(data_all)

# 带阴影的绘图函数
def func_plot_shaded(data_matrix, label, color):
    num_data_points = data_matrix.shape[1]
    x = np.linspace(0, num_data_points, num_data_points)
    mean = data_matrix.mean(axis=0)
    std = data_matrix.std(axis=0)

    # 绘制线条和阴影
    plt.plot(x, mean, label=label, linewidth=1, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def func_plot_shaded_with_ma(data, label, color, window_size):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ma_mean = moving_average(mean, window_size)
    std = std[window_size-1:]
    x = np.arange(len(ma_mean))
    plt.fill_between(x, ma_mean - std, ma_mean + std, alpha=0.1, color=color)
    plt.plot(x, ma_mean, label=label, color=color, linewidth=1)


# 加载数据
num_runs_begin = 0
num_runs_end   = 5

# print('\n---------------------------------------- CO sparse ----------------------------------------\n')

ppobc_sparse  = load_data('./data/toy_ppobc_sparse_{}.npy',  num_runs_begin, num_runs_end)
causal_sparse = load_data('./data/toy_causal_sparse_{}.npy', num_runs_begin, num_runs_end)
bcr_sparse    = load_data('./data/toy_bcr_sparse_{}.npy', num_runs_begin, num_runs_end)
# 创建图形并设置背景
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=100)
fig.patch.set_facecolor('white')  # 整个图表背景为白色
ax.set_facecolor('#e6e6fa')  # 仅设置框框内为浅紫色
ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.8)  # 白色网格线
# 绘制曲线
window_size = 4
func_plot_shaded_with_ma(ppobc_sparse,  label=r'PPO$_{\mathrm{BC}}$', color='#1f77b4', window_size=window_size)  # 蓝色
func_plot_shaded_with_ma(causal_sparse, label='Causal',               color='#2ca02c', window_size=window_size)  # 绿色
func_plot_shaded_with_ma(bcr_sparse,    label=r'BCR-DRL',             color='#d62728', window_size=window_size)  # 红色
# 设置坐标标签和范围
plt.xlabel('Epoch', fontsize=fontsize_figure, color='black')
plt.ylabel('Average accmulated reward', fontsize=fontsize_figure, color='black')
plt.tick_params(labelsize=fontsize_figure, colors='black')  # 坐标轴刻度颜色为黑色
# plt.xlim([0, xlim_max])
plt.ylim([0, 600])
# # 调整刻度间隔
# plt.xticks(np.arange(0, xlim_max + 1, 100))  # X轴每隔50一个刻度
# plt.yticks(np.arange(0, ylim_max + 1, 50))  # Y轴每隔50一个刻度
# 将图例放置在框框的上面
plt.legend(loc='upper center', fontsize=fontsize_figure, ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15))
# 设置外框的颜色为白色
for spine in ax.spines.values():
    spine.set_edgecolor('white')  # 设置外框线条为白色
# 调整布局增加边距
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
plt.savefig('./figs/CR_ablation.pdf', format='pdf')








# fontsize_figure = 14  # 字体大小
# xlim_max = 400
# ylim_max = 200

# # 数据加载函数（可以替换为你的实际数据加载逻辑）
# def load_data(file_pattern, num_runs_begin, num_runs_end):
#     data_all = []
#     for index_run in range(num_runs_begin, num_runs_end):
#         data_all.append(np.load(file_pattern.format(index_run)))
#     return np.array(data_all)

# # 带阴影的绘图函数
# def func_plot_shaded(data_matrix, label, color):
#     num_data_points = data_matrix.shape[1]
#     x = np.linspace(0, num_data_points, num_data_points)
#     mean = data_matrix.mean(axis=0)
#     std = data_matrix.std(axis=0)

#     # 绘制线条和阴影
#     plt.plot(x, mean, label=label, linewidth=1, color=color)
#     plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

# # 加载数据
# num_runs_begin = 0
# num_runs_end = 5


# # print('\n---------------------------------------- CR ablation no weights ----------------------------------------\n')

# ppobc_sparse  = load_data('./data/toy_ppobc_sparse_{}.npy',  num_runs_begin, num_runs_end)
# # causal_sparse = load_data('./data/toy_causal_sparse_{}.npy', num_runs_begin, num_runs_end)
# bcr_sparse    = load_data('./data/toy_bcr_sparse_{}.npy', num_runs_begin, num_runs_end)


# # 创建图形并设置背景
# fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=100)
# fig.patch.set_facecolor('white')  # 整个图表背景为白色
# ax.set_facecolor('#e6e6fa')  # 仅设置框框内为浅紫色
# ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.8)  # 白色网格线
# # 绘制曲线
# func_plot_shaded(ppobc_sparse,  label=r'PPO$_{\mathrm{BC}}$', color='#1f77b4')  # 蓝色
# # func_plot_shaded(causal_sparse, label='Causal', color='#2ca02c')               # 绿色
# func_plot_shaded(bcr_sparse,    label=r'BCR-DRL', color='#d62728')                # 红色
# # 设置坐标标签和范围
# plt.xlabel('Epoch', fontsize=fontsize_figure, color='black')
# plt.ylabel('Sparse return', fontsize=fontsize_figure, color='black')
# plt.tick_params(labelsize=fontsize_figure, colors='black')  # 坐标轴刻度颜色为黑色
# # plt.xlim([0, xlim_max])
# # plt.ylim([0, ylim_max])
# # 调整刻度间隔
# # plt.xticks(np.arange(0, xlim_max + 1, 100))  # X轴每隔50一个刻度
# # plt.yticks(np.arange(0, ylim_max + 1, 50))  # Y轴每隔50一个刻度
# # 将图例放置在框框的上面
# plt.legend(loc='upper center', fontsize=fontsize_figure, ncol=1, frameon=False, bbox_to_anchor=(0.7, 0.4))
# # 设置外框的颜色为白色
# for spine in ax.spines.values():
#     spine.set_edgecolor('white')  # 设置外框线条为白色
# # 调整布局增加边距
# fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# # 保存和显示图表


plt.show()