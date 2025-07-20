from BC_model_functions import load_bc_model

bc_model_path = "./bc_runs_cimar/reproduce/cramped_room"
bc_model, bc_params = load_bc_model(bc_model_path)
bc_model.summary()
print('bc_params:', bc_params)










# from BC_model_functions_load import _get_base_ae, BehaviorCloningPolicy
# bc_policy = BehaviorCloningPolicy.from_model(bc_model, bc_params, stochastic=True)

# base_ae = _get_base_ae(bc_params)
# base_env = base_ae.env

# from BC_model_functions import RlLibAgent
# bc_agent = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)
# bc_agent

