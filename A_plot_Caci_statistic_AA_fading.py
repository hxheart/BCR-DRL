import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


fontsize_sparse_figure = 12
buttom_sparse_figure = 0.15

num_index_run = 18
xlim_max=int(400)
ylim_max=int(220)




# ################ training ###############################################################################################


def func_plot_shaded(data_matrix, label, color=None): # 艾玛，用标准差的
    num_data_points = data_matrix.shape[1]
    x = np.linspace(0, num_data_points, num_data_points)  # 这里假设总环境交互次数从0到num_data_points

    mean = data_matrix.mean(axis=0)
    std = data_matrix.std(axis=0)

    line = plt.plot(x, mean, label=label, linewidth=0.5, color=color)[0]
    if color == None:
        color = line.get_color()
    else:
        color = plt.matplotlib.colors.to_rgba(color)

    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    plt.legend()


# ====================== fading in AA ==================================================



# print('\n---------------------------------------- fading ireca solo ----------------------------------------\n')

ireca_return_shaped_all= []
ireca_return_sparse_all= []
ireca_return_env_all= []
ireca_return_cau_all= []
ireca_return_ent_all= []
ireca_return_ireca_all= []

for index_run in range(0, num_index_run, 1):
    ireca_return_shaped_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_shaped_{index_run}.npy'))
    ireca_return_sparse_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_sparse_{index_run}.npy'))
    ireca_return_env_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_cau_{index_run}.npy'))
    ireca_return_cau_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_cau_{index_run}.npy'))
    ireca_return_ent_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_ent_{index_run}.npy'))
    ireca_return_ireca_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_ireca_{index_run}.npy'))


ireca_return_shaped_all = np.array(ireca_return_shaped_all)
ireca_return_sparse_all = np.array(ireca_return_sparse_all)
ireca_return_env_all = np.array(ireca_return_env_all)
ireca_return_cau_all = np.array(ireca_return_cau_all)
ireca_return_ent_all = np.array(ireca_return_ent_all)
ireca_return_ireca_all = np.array(ireca_return_ireca_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ireca_return_sparse_all, label= 'Sparse', color='C1')
func_plot_shaded(ireca_return_shaped_all, label= 'Stage',  color='C5')
# func_plot_shaded(ireca_return_env_all,    'Env')
func_plot_shaded(ireca_return_cau_all,    label= r'Intrinsic: $\bar{R}^{\mathcal{I}_\mathrm{H}}$',   color='C6')
func_plot_shaded(ireca_return_ent_all,    label= r'Intrinsic: $\bar{R}^{\mathcal{I}_\mathrm{A}}$',   color='C14')
# func_plot_shaded(ireca_return_ireca_all,  label='Sum')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Return of IReCa', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/aa_fading_return_ireca.pdf', format='pdf')



# # # print('\n---------------------------------------- fading causal solo ----------------------------------------\n')
causal_return_shaped_all= []
causal_return_sparse_all= []
causal_return_env_all= []
causal_return_cau_all= []
causal_return_causal_all= []

for index_run in range(0, num_index_run, 1):
    causal_return_shaped_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_shaped_{index_run}.npy'))
    causal_return_sparse_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_sparse_{index_run}.npy'))
    causal_return_env_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_env_{index_run}.npy'))
    causal_return_cau_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_cau_{index_run}.npy'))
    causal_return_causal_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_causal_{index_run}.npy'))

causal_return_shaped_all = np.array(causal_return_shaped_all)
causal_return_sparse_all = np.array(causal_return_sparse_all)
causal_return_env_all = np.array(causal_return_env_all)
causal_return_cau_all = np.array(causal_return_cau_all)
causal_return_causal_all = np.array(causal_return_causal_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(causal_return_sparse_all,label= 'Sparse', color='C1')
func_plot_shaded(causal_return_shaped_all, label='Stage',  color='C5')
# func_plot_shaded(causal_return_env_all,    'Env')
func_plot_shaded(causal_return_cau_all,    label= 'Causal',color='C6')
# func_plot_shaded(causal_return_causal_all,  label='Sum')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Return of Causal', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/aa_fading_return_causal.pdf', format='pdf')


# # # print('\n---------------------------------------- fading ppobc solo ----------------------------------------\n')
ppobc_return_shaped_all= []
ppobc_return_sparse_all= []
ppobc_return_env_all= []

for index_run in range(0, num_index_run, 1):
    ppobc_return_shaped_all.append(np.load(f'./data/aa_ppobc/aa_data_fading_ppobc_return_shaped_{index_run}.npy'))
    ppobc_return_sparse_all.append(np.load(f'./data/aa_ppobc/aa_data_fading_ppobc_return_sparse_{index_run}.npy'))
    # ppobc_return_env_all.append(np.load(f'./data/aa_ppobc/aa_data_fading_ppobc_return_env_{index_run}.npy'))

ppobc_return_shaped_all = np.array(ppobc_return_shaped_all)
ppobc_return_sparse_all = np.array(ppobc_return_sparse_all)
# ppobc_return_env_all = np.array(ppobc_return_env_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ppobc_return_sparse_all,label= 'Sparse', color='C1')
func_plot_shaded(ppobc_return_shaped_all, label='Stage',  color='C5')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel(r'Return of PPO$_{\mathrm{BC}}$', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/aa_fading_return_ppobc.pdf', format='pdf')

# print('\n---------------------------------------- (training) aa fading compare sparse ----------------------------------------\n')

ppobc_return_sparse_all= []
causal_return_sparse_all= []
ireca_return_sparse_all= []

for index_run in range(0, num_index_run, 1):
    ppobc_return_sparse_all.append(np.load(f'./data/aa_ppobc/aa_data_fading_ppobc_return_sparse_{index_run}.npy'))
    causal_return_sparse_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_sparse_{index_run}.npy'))
    ireca_return_sparse_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_sparse_{index_run}.npy'))

ppobc_return_sparse_all = np.array(ppobc_return_sparse_all)
causal_return_sparse_all = np.array(causal_return_sparse_all)
ireca_return_sparse_all = np.array(ireca_return_sparse_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ppobc_return_sparse_all, label= r'PPO$_{\mathrm{BC}}$', color='m')
func_plot_shaded(causal_return_sparse_all, label= 'Causal',               color='g')
func_plot_shaded(ireca_return_sparse_all, label= r'IReCa',                color='r')
plt.xlabel('Epoch', fontsize = fontsize_sparse_figure)
plt.ylabel('Sparse return', fontsize = fontsize_sparse_figure)
plt.legend(loc='best', fontsize = fontsize_sparse_figure, ncol=1) 
plt.tick_params(labelsize = fontsize_sparse_figure)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/aa_fading_compare_sparse_return.pdf', format='pdf')




# print('\n---------------------------------------- (training) aa fading compare stage ----------------------------------------\n')
ppobc_return_shaped_all= []
causal_return_shaped_all= []
ireca_return_shaped_all= []

for index_run in range(0, num_index_run, 1):
    ppobc_return_shaped_all.append(np.load(f'./data/aa_ppobc/aa_data_fading_ppobc_return_shaped_{index_run}.npy'))
    causal_return_shaped_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_shaped_{index_run}.npy'))
    ireca_return_shaped_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_shaped_{index_run}.npy'))

ppobc_return_shaped_all = np.array(ppobc_return_shaped_all)
causal_return_shaped_all = np.array(causal_return_shaped_all)
ireca_return_shaped_all = np.array(ireca_return_shaped_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ppobc_return_shaped_all, label= r'PPO$_{\mathrm{BC}}$', color='m')
func_plot_shaded(causal_return_shaped_all, label= 'Causal',               color='g')
func_plot_shaded(ireca_return_shaped_all, label= r'IReCa',                color='r')

plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Stage return', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/aa_fading_compare_shaped_return.pdf', format='pdf')




# print('\n---------------------------------------- (training) aa fading compare env ----------------------------------------\n')
ppobc_return_env_all= []
causal_return_env_all= []
ireca_return_env_all= []

for index_run in range(0, num_index_run, 1):
    ppobc_return_env_all.append(np.load(f'./data/aa_ppobc/aa_data_fading_ppobc_return_env_{index_run}.npy'))
    causal_return_env_all.append(np.load(f'./data/aa_causal/aa_data_fading_causal_return_env_{index_run}.npy'))
    ireca_return_env_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_return_env_{index_run}.npy'))

ppobc_return_env_all = np.array(ppobc_return_env_all)
causal_return_env_all = np.array(causal_return_env_all)
ireca_return_env_all = np.array(ireca_return_env_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ppobc_return_env_all, label= r'PPO$_{\mathrm{BC}}$', color='m')
func_plot_shaded(causal_return_env_all, label= 'Causal',               color='g')
func_plot_shaded(ireca_return_env_all, label= r'IReCa',                color='r')

plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Extrinsic return', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, xlim_max])
plt.ylim([0, ylim_max])
plt.savefig('./figs/aa_fading_compare_env_return.pdf', format='pdf')

# print('\n---------------------------------------- fading weights solo ----------------------------------------\n')

ireca_return_softmax_env_all= []

for index_run in range(0, num_index_run, 1):
    ireca_return_softmax_env_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_softmax_env_{index_run}.npy'))
ireca_return_softmax_env_all = np.array(ireca_return_softmax_env_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ireca_return_softmax_env_all, label= r'${\kappa}_{n}^\mathcal{E}$', color='C7')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Context-aware weight', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, 200])
plt.ylim([-0.1, 2])
plt.savefig('./figs/aa_fading_compare_softmax_env_return.pdf', format='pdf')

# print('\n---------------------------------------- fading weights solo ----------------------------------------\n')
ireca_return_softmax_ent_all= []

for index_run in range(0, num_index_run, 1):
    ireca_return_softmax_ent_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_softmax_ent_{index_run}.npy'))
ireca_return_softmax_ent_all = np.array(ireca_return_softmax_ent_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ireca_return_softmax_ent_all, label= r'${\kappa}_{n}^{\mathcal{I}_{\mathrm{A}}}$', color='C9')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Context-aware weight', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, 200])
plt.ylim([-0.1, 2])
plt.savefig('./figs/aa_fading_compare_softmax_ent_return.pdf', format='pdf')


# print('\n---------------------------------------- fading weights solo ----------------------------------------\n')

ireca_return_softmax_cau_all= []

for index_run in range(0, num_index_run, 1):
    ireca_return_softmax_cau_all.append(np.load(f'./data/aa_ireca/aa_data_fading_ireca_softmax_cau_{index_run}.npy'))
ireca_return_softmax_cau_all = np.array(ireca_return_softmax_cau_all)

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.2), dpi=70)
func_plot_shaded(ireca_return_softmax_cau_all, label= r'${\kappa}_{n}^{\mathcal{I}_{\mathrm{H}}}$', color='C10')
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Context-aware weight', fontsize = 12)
plt.legend(loc='best', fontsize = 12, ncol=1) 
plt.tick_params(labelsize = 12)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95, hspace=1, wspace=1)
plt.xlim([0, 200])
plt.ylim([-0.1, 2])
plt.savefig('./figs/aa_fading_compare_softmax_cau_return.pdf', format='pdf')

# plt.show()

# print('\n---------------------------------------- (testing aa)----------------------------------------\n')

# 读取数据
aa_ppobc_return_sparse_mean = np.mean(np.load(f'./data/test/aa_ppobc/aa_test_data_fading_ppobc_return_sparse_0.npy'))
aa_causal_return_sparse_mean = np.mean(np.load(f'./data/test/aa_causal/aa_test_data_fading_causal_return_sparse_0.npy'))
aa_ireca_return_sparse_mean = np.mean(np.load(f'./data/test/aa_ireca/aa_test_data_fading_ireca_return_sparse_0.npy'))

# 设置任务和模型
tasks = ["Sparse return"]
models = ["PPO$_{\mathrm{BC}}$", "Causal", "IReCa"]
data = [aa_ppobc_return_sparse_mean, aa_causal_return_sparse_mean, aa_ireca_return_sparse_mean]

# 设置条形图参数
width = 0.05  # 条形的宽度
index = np.arange(len(tasks))  # 任务的 x 轴位置
center_positions = index + width * (len(models) - 1) / 2
colors = ['m', 'g', 'r']

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(2, 3.2), dpi=70)
for i, model_data in enumerate(data):
    color = colors[i]
    ax.bar(index + i * width, [model_data], width, label=models[i], color=color)

# 设置 x 轴刻度和标签
ax.set_xticks(center_positions)
ax.set_xticklabels(tasks)
plt.tick_params(labelsize=12)

# 调整子图的间距
fig.subplots_adjust(left=0.3, bottom=0.16, right=0.95, top=0.95)

# 设置 y 轴限制和标签
plt.ylim([0, ylim_max])
# plt.ylabel('Average episode sparse return', fontsize=12)
# ax.legend(loc='upper right', fontsize=12, ncol=1)
# 保存图形
plt.savefig('./figs/test_aa_sparse_fading_compare_return.pdf', format='pdf')


# print('\n---------------------------------------- (testing aa stage)----------------------------------------\n')
aa_ppobc_return_sparse_mean = np.mean(np.load(f'./data/test/aa_ppobc/aa_test_data_fading_ppobc_return_shaped_0.npy'))
aa_causal_return_sparse_mean = np.mean(np.load(f'./data/test/aa_causal/aa_test_data_fading_causal_return_shaped_0.npy'))
aa_ireca_return_sparse_mean = np.mean(np.load(f'./data/test/aa_ireca/aa_test_data_fading_ireca_return_shaped_0.npy'))

# 设置任务和模型
tasks = ["Sparse return"]
models = ["PPO$_{\mathrm{BC}}$", "Causal", "IReCa"]
data = [aa_ppobc_return_sparse_mean, aa_causal_return_sparse_mean, aa_ireca_return_sparse_mean]

# 设置条形图参数
width = 0.05  # 条形的宽度
index = np.arange(len(tasks))  # 任务的 x 轴位置
center_positions = index + width * (len(models) - 1) / 2
colors = ['m', 'g', 'r']

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(2, 3.2), dpi=70)
for i, model_data in enumerate(data):
    color = colors[i]
    ax.bar(index + i * width, [model_data], width, label=models[i], color=color)

# 设置 x 轴刻度和标签
ax.set_xticks(center_positions)
ax.set_xticklabels(tasks)
plt.tick_params(labelsize=12)

# 调整子图的间距
fig.subplots_adjust(left=0.3, bottom=0.16, right=0.95, top=0.95)

# 设置 y 轴限制和标签
plt.ylim([0, ylim_max])
# plt.ylabel('Average episode stage return', fontsize=12)
# ax.legend(loc='upper right', fontsize=12, ncol=1)
# 保存图形
plt.savefig('./figs/test_aa_shaped_fading_compare_return.pdf', format='pdf')



plt.show()

