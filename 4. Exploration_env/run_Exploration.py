from HyperParameters import *

num_train_begin = 3
num_train_end = 5


for index_run in range(num_train_begin, num_train_end):
    print('------------------> index run:', index_run)
    with open('Explore_ppobc_20250430.py', 'r', encoding='utf-8') as file:
        exec(file.read())
    np.save(f'./data/toy_ppobc_sparse_{index_run}.npy', accumulated_sparse)

# # # --------------------------------------

# for index_run in range(num_train_begin, num_train_end):
#     print('------------------> index run:', index_run)
#     with open('Explore_Causal_20250430.py', 'r', encoding='utf-8') as file:
#         exec(file.read())
#     np.save(f'./data/toy_causal_sparse_{index_run}.npy', accumulated_sparse)

# --------------------------------------

# for index_run in range(num_train_begin, num_train_end):
#     print('------------------> index run:', index_run)
#     with open('Explore_bcr_20250430.py', 'r', encoding='utf-8') as file:
#         exec(file.read())
#     np.save(f'./data/toy_bcr_sparse_{index_run}.npy', accumulated_sparse)


plt.show()