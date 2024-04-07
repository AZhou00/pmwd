import pickle
from pmwd.sto.mlp import mlp_size

def load_soparams(so_params):
    if isinstance(so_params, str):
        with open(so_params, 'rb') as f:
            so_params = pickle.load(f)['so_params']
    n_input, so_nodes = mlp_size(so_params)
    return so_params, n_input, so_nodes
