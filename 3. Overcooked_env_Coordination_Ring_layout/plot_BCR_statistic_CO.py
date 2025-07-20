import matplotlib.pyplot as plt
import numpy as np

fontsize_figure = 14  # 字体大小
xlim_max = 600
ylim_max = 150

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
    plt.plot(x, mean, label=label, linewidth=0.5, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def func_plot_shaded_with_ma(data, label, color, window_size):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ma_mean = moving_average(mean, window_size)
    std = std[window_size-1:]
    x = np.arange(len(ma_mean))
    plt.fill_between(x, ma_mean - std, ma_mean + std, alpha=0.1, color=color)
    plt.plot(x, ma_mean, label=label, color=color, linewidth=0.5)


# 加载数据
num_runs_begin = 0
num_runs_end   = 5

# print('\n---------------------------------------- CO sparse ----------------------------------------\n')

ppobc_return_sparse_all = load_data('./data/co_ppobc/co_data_ppobc_return_sparse_{}.npy', num_runs_begin, num_runs_end)
cippo_return_sparse_all = load_data('./data/co_causal/co_data_causal_return_sparse_{}.npy', num_runs_begin, num_runs_end)
caci_return_sparse_all  = load_data('./data/co_bcr/co_data_bcr_return_sparse_{}.npy',   num_runs_begin, num_runs_end)
# 创建图形并设置背景
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=100)
fig.patch.set_facecolor('white')  # 整个图表背景为白色
ax.set_facecolor('#e6e6fa')  # 仅设置框框内为浅紫色
ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.8)  # 白色网格线
# 绘制曲线
func_plot_shaded_with_ma(ppobc_return_sparse_all, label=r'PPO$_{\mathrm{BC}}$', color='#1f77b4', window_size=1)  # 蓝色
func_plot_shaded_with_ma(cippo_return_sparse_all, label='Causal',               color='#2ca02c', window_size=1)  # 绿色
func_plot_shaded_with_ma(caci_return_sparse_all,  label=r'BCD',                 color='#d62728', window_size=1)  # 红色
# 设置坐标标签和范围
plt.xlabel('Epoch', fontsize=fontsize_figure, color='black')
plt.ylabel('Sparse return', fontsize=fontsize_figure, color='black')
plt.tick_params(labelsize=fontsize_figure, colors='black')  # 坐标轴刻度颜色为黑色
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
# 调整刻度间隔
plt.xticks(np.arange(0, xlim_max + 1, 100))  # X轴每隔50一个刻度
plt.yticks(np.arange(0, ylim_max + 1, 50))  # Y轴每隔50一个刻度
# 将图例放置在框框的上面
plt.legend(loc='upper center', fontsize=fontsize_figure, ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15))
# 设置外框的颜色为白色
for spine in ax.spines.values():
    spine.set_edgecolor('white')  # 设置外框线条为白色
# 调整布局增加边距
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.85)
# 保存和显示图表
plt.savefig('./figs/co_fading_compare_sparse_return.pdf', format='pdf')



# print('\n---------------------------------------- sparse testing results----------------------------------------\n')

co_ppobc_return_sparse_mean = np.mean(np.load(f'./data/test/co_ppobc/co_test_data_ppobc_return_sparse.npy'))
co_cippo_return_sparse_mean = np.mean(np.load(f'./data/test/co_causal/co_test_data_causal_return_sparse.npy'))
co_bcr_return_sparse_mean = np.mean(np.load(f'./data/test/co_bcr/co_test_data_bcr_return_sparse.npy'))
data = [co_ppobc_return_sparse_mean, co_cippo_return_sparse_mean, co_bcr_return_sparse_mean]

co_ppobc_return_sparse_std = np.std(np.load(f'./data/test/co_ppobc/co_test_data_ppobc_return_sparse.npy'))
co_cippo_return_sparse_std = np.std(np.load(f'./data/test/co_causal/co_test_data_causal_return_sparse.npy'))
co_bcr_return_sparse_std = np.std(np.load(f'./data/test/co_bcr/co_test_data_bcr_return_sparse.npy'))
std_data = [co_ppobc_return_sparse_std, co_cippo_return_sparse_std, co_bcr_return_sparse_std]

tasks = ["Sparse return"]
models = ["PPO$_{\mathrm{BC}}$", "Causal", "BCR-DRL"]
width = 0.05  # 条形的宽度
index = np.arange(len(tasks))  # 任务的 x 轴位置
center_positions = index + width * (len(models) - 1) / 2
colors = ['#1f77b4', '#d62728', '#2ca02c']
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=100)
for i, (model_data, model_std) in enumerate(zip(data, std_data)):
    color = colors[i]
    ax.bar(index + i * width, [model_data], width, label=models[i], color=color, yerr=[model_std], capsize=5)
ax.set_xticks(center_positions)
ax.set_xticklabels(tasks)
plt.tick_params(labelsize=12)
fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.85)
plt.ylim([0, ylim_max])
plt.ylabel('Average episode sparse return', fontsize=12)
ax.legend(loc='upper right', fontsize=12, ncol=1)
plt.legend(loc='upper center', fontsize=fontsize_figure, ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15))
plt.savefig('./figs/test_co_sparse_fading_compare_return.pdf', format='pdf')

# plt.show()


# # # print('\n---------------------------------------- ppo_bc rewards ----------------------------------------\n')
ppobc_return_shaped_all= []
ppobc_return_sparse_all= []


for index_run in range(num_runs_begin, num_runs_end):
    ppobc_return_shaped_all.append(np.load(f'./data/co_ppobc/co_data_ppobc_return_stage_{index_run}.npy'))
    ppobc_return_sparse_all.append(np.load(f'./data/co_ppobc/co_data_ppobc_return_sparse_{index_run}.npy'))

ppobc_return_shaped_all = np.array(ppobc_return_shaped_all)
ppobc_return_sparse_all = np.array(ppobc_return_sparse_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ppobc_return_sparse_all,label= 'Sparse', color='C1')
func_plot_shaded(ppobc_return_shaped_all, label='Stage',  color='C5')
# func_plot_shaded(ppobc_return_env_all, label='Env',  color='C6')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel(r'Return of PPO$_{\mathrm{BC}}$', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/co_fading_return_ppobc.pdf', format='pdf')





# # # # print('\n---------------------------------------- causal rewards ----------------------------------------\n')
cippo_return_shaped_all= []
cippo_return_sparse_all= []
cippo_return_env_all= []
cippo_return_cau_all= []
cippo_return_cippo_all= []

for index_run in range(num_runs_begin, num_runs_end):
    cippo_return_shaped_all.append(np.load(f'./data/co_causal/co_data_causal_return_stage_{index_run}.npy'))
    cippo_return_sparse_all.append(np.load(f'./data/co_causal/co_data_causal_return_sparse_{index_run}.npy'))
    cippo_return_cau_all.append(np.load(f'./data/co_causal/co_data_causal_return_causal_{index_run}.npy'))

cippo_return_shaped_all = np.array(cippo_return_shaped_all)
cippo_return_sparse_all = np.array(cippo_return_sparse_all)
cippo_return_cau_all = np.array(cippo_return_cau_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(cippo_return_sparse_all,label= 'Sparse', color='C1')
func_plot_shaded(cippo_return_shaped_all, label='Stage',  color='C5')
func_plot_shaded(cippo_return_cau_all,    label= 'Causal',color='C6')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Return of Causal', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
# plt.ylim([0, ylim_max])
plt.savefig('./figs/co_fading_return_cippo.pdf', format='pdf')

# print('\n---------------------------------------- caci rewards ----------------------------------------\n')

caci_return_shaped_all= []
caci_return_sparse_all= []
caci_return_env_all= []
caci_return_cau_all= []
caci_return_ent_all= []
caci_return_caci_all= []

for index_run in range(num_runs_begin, num_runs_end):
    caci_return_shaped_all.append(np.load(f'./data/co_bcr/co_data_bcr_return_stage_{index_run}.npy'))
    caci_return_sparse_all.append(np.load(f'./data/co_bcr/co_data_bcr_return_sparse_{index_run}.npy'))
    caci_return_cau_all.append(np.load(f'./data/co_bcr/cr_data_bcr_return_human_{index_run}.npy'))
    caci_return_ent_all.append(np.load(f'./data/co_bcr/cr_data_bcr_return_AI_{index_run}.npy'))


caci_return_shaped_all = np.array(caci_return_shaped_all)
caci_return_sparse_all = np.array(caci_return_sparse_all)
caci_return_cau_all = np.array(caci_return_cau_all)
caci_return_ent_all = np.array(caci_return_ent_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(caci_return_sparse_all, label= 'Sparse', color='C1')
func_plot_shaded(caci_return_shaped_all, label= 'Stage',  color='C5')
func_plot_shaded(caci_return_cau_all,    label= r'Intrinsic: $\bar{R}^{\mathcal{I}_\mathrm{H}}$',   color='C6')
func_plot_shaded(caci_return_ent_all,    label= r'Intrinsic: $\bar{R}^{\mathcal{I}_\mathrm{A}}$',   color='C14')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Return of IReCa', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/co_fading_return_caci.pdf', format='pdf')

# plt.show()



# print('\n---------------------------------------- weights extrinsic ----------------------------------------\n')

caci_return_softmax_env_all= []

for index_run in range(num_runs_begin, num_runs_end):
    caci_return_softmax_env_all.append(np.load(f'./data/co_bcr/co_data_bcr_weights_env_{index_run}.npy'))
caci_return_softmax_env_all = np.array(caci_return_softmax_env_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(caci_return_softmax_env_all, label= r'${\kappa}_{n}^\mathcal{E}$', color='C7')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Context-aware weight', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, 400])
plt.ylim([0.5, 1.5])
plt.savefig('./figs/co_fading_compare_softmax_env_return.pdf', format='pdf')

# print('\n---------------------------------------- weights intrinsic AI ----------------------------------------\n')
caci_return_softmax_ent_all= []

for index_run in range(num_runs_begin,num_runs_end):
    caci_return_softmax_ent_all.append(np.load(f'./data/co_bcr/co_data_bcr_weights_AI_{index_run}.npy'))
caci_return_softmax_ent_all = np.array(caci_return_softmax_ent_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(caci_return_softmax_ent_all, label= r'${\kappa}_{n}^{\mathcal{I}_{\mathrm{A}}}$', color='C9')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Context-aware weight', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, 400])
plt.ylim([-0.1, 1.5])
plt.savefig('./figs/co_fading_compare_softmax_ent_return.pdf', format='pdf')


# print('\n---------------------------------------- weights intrinsic HM ----------------------------------------\n')

caci_return_softmax_cau_all= []

for index_run in range(num_runs_begin, num_runs_end):
    caci_return_softmax_cau_all.append(np.load(f'./data/co_bcr/co_data_bcr_weights_human_{index_run}.npy'))
caci_return_softmax_cau_all = np.array(caci_return_softmax_cau_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(caci_return_softmax_cau_all, label= r'${\kappa}_{n}^{\mathcal{I}_{\mathrm{H}}}$', color='C10')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Context-aware weight', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, 400])
plt.ylim([-0.1, 1.5])
plt.savefig('./figs/co_fading_compare_softmax_cau_return.pdf', format='pdf')



# print('\n---------------------------------------- CO ablation no weights ----------------------------------------\n')
fontsize_figure = 14  # 字体大小
xlim_max = 600
ylim_max = 150

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
    plt.plot(x, ma_mean, label=label, color=color, linewidth=0.5)


# 加载数据
num_runs_begin = 0
num_runs_end   = 5

ppobc_return_sparse_all = load_data('./data/co_ppobc/co_data_ppobc_return_sparse_{}.npy', num_runs_begin, num_runs_end)
caci_no_weights_return_sparse_all = load_data('./data/co_bcr_no_weights/co_data_bcr_no_weights_return_sparse_{}.npy', num_runs_begin, num_runs_end)
caci_return_sparse_all  = load_data('./data/co_bcr/co_data_bcr_return_sparse_{}.npy',   num_runs_begin, num_runs_end)


# 创建图形并设置背景
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), dpi=100)
fig.patch.set_facecolor('white')  # 整个图表背景为白色
ax.set_facecolor('#e6e6fa')  # 仅设置框框内为浅紫色
ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.8)  # 白色网格线
# 绘制曲线
func_plot_shaded(ppobc_return_sparse_all, label=r'BCR-NoIntrinsic', color='#1f77b4')  # 蓝色
func_plot_shaded(caci_no_weights_return_sparse_all, label='BCR-NoWeights', color='#2ca02c')               # 绿色
func_plot_shaded(caci_return_sparse_all, label=r'BCR-DRL', color='#d62728')                # 红色
# 设置坐标标签和范围
plt.xlabel('Epoch', fontsize=fontsize_figure, color='black')
plt.ylabel('Sparse return', fontsize=fontsize_figure, color='black')
plt.tick_params(labelsize=fontsize_figure, colors='black')  # 坐标轴刻度颜色为黑色
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
# 调整刻度间隔
plt.xticks(np.arange(0, xlim_max + 1, 100))  # X轴每隔50一个刻度
plt.yticks(np.arange(0, ylim_max + 1, 50))  # Y轴每隔50一个刻度
# 将图例放置在框框的上面
plt.legend(loc='upper center', fontsize=fontsize_figure, ncol=1, frameon=False, bbox_to_anchor=(0.75, 0.35))
# 设置外框的颜色为白色
for spine in ax.spines.values():
    spine.set_edgecolor('white')  # 设置外框线条为白色
# 调整布局增加边距
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
# 保存和显示图表
plt.savefig('./figs/CO_ablation.pdf', format='pdf')

plt.show()

























plt.show()