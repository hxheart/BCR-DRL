import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用哪块GPU

from HyperParameters import *


if learning_rate_reward_shaping == 4e-6:
    for index_run in range(0,1):
        print('------------------> index run:', index_run)
        # ----------------------
        if bc_model_path_test == "./bc_runs_ireca/reproduce_test/cramped_room": 
            with open('test_ppobc.py', 'r', encoding='utf-8') as file:
                exec(file.read())
            print(f'bc_model_path_test: {bc_model_path_test}') 
            np.save(f'./data/test/cr_ppobc/rc_test_data_fading_ppobc_return_shaped_{index_run}.npy', avg_return_shaped)
            np.save(f'./data/test/cr_ppobc/rc_test_data_fading_ppobc_return_sparse_{index_run}.npy', avg_return_sparse)
            np.save(f'./data/test/cr_ppobc/rc_test_data_fading_ppobc_return_env_{index_run}.npy',    avg_return_env)
        elif bc_model_path_test == "./bc_runs_ireca/reproduce_test/asymmetric_advantages": 
            with open('test_ppobc.py', 'r', encoding='utf-8') as file:
                exec(file.read())
            print(f'bc_model_path_test: {bc_model_path_test}') 
            np.save(f'./data/test/aa_ppobc/aa_test_data_fading_ppobc_return_shaped_{index_run}.npy', avg_return_shaped)
            np.save(f'./data/test/aa_ppobc/aa_test_data_fading_ppobc_return_sparse_{index_run}.npy', avg_return_sparse)
            np.save(f'./data/test/aa_ppobc/aa_test_data_fading_ppobc_return_env_{index_run}.npy',    avg_return_env)


