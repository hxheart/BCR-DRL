import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用哪块GPU

from HyperParameters import *


if learning_rate_reward_shaping == 4e-6:
    for index_run in range(16,18):
        print('------------------> index run:', index_run)
        if bc_model_path_train == "./bc_runs_ireca/reproduce_train/cramped_room":
            print(f'bc_model_path_train: {bc_model_path_train}')
            with open('train_causal.py', 'r', encoding='utf-8') as file:
                exec(file.read())
            # ----------------------
            np.save(f'./data_tmp/cr_causal/rc_data_fading_causal_return_shaped_{index_run}.npy', avg_return_shaped)
            np.save(f'./data_tmp/cr_causal/rc_data_fading_causal_return_sparse_{index_run}.npy', avg_return_sparse)
            np.save(f'./data_tmp/cr_causal/rc_data_fading_causal_return_env_{index_run}.npy',    avg_return_env)
            np.save(f'./data_tmp/cr_causal/rc_data_fading_causal_return_cau_{index_run}.npy',    avg_return_cau)
            np.save(f'./data_tmp/cr_causal/rc_data_fading_causal_return_causal_{index_run}.npy',  avg_return_causal)
        # --
        elif bc_model_path_train == "./bc_runs_ireca/reproduce_train/asymmetric_advantages":
            print(f'bc_model_path_train: {bc_model_path_train}')
            with open('train_causal.py', 'r', encoding='utf-8') as file:
                exec(file.read())
            # ----------------------
            np.save(f'./data_tmp/aa_causal/aa_data_fading_causal_return_shaped_{index_run}.npy', avg_return_shaped)
            np.save(f'./data_tmp/aa_causal/aa_data_fading_causal_return_sparse_{index_run}.npy', avg_return_sparse)
            np.save(f'./data_tmp/aa_causal/aa_data_fading_causal_return_env_{index_run}.npy',    avg_return_env)
            np.save(f'./data_tmp/aa_causal/aa_data_fading_causal_return_cau_{index_run}.npy',    avg_return_cau)
            np.save(f'./data_tmp/aa_causal/aa_data_fading_causal_return_causal_{index_run}.npy',  avg_return_causal)







