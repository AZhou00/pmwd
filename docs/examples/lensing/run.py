import os
import sys
import json
import argparse

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
sys.path.insert(0, '/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/fin_test')

import jax.numpy as jnp
import numpy as np
from pmwd import (
    Configuration,
    boltzmann,
    Cosmology,
    white_noise,
    linear_modes,
    lpt,
    nbody,
    nbody_ray,
    Rays,
    lens_plane_diagram,
    load_soparams
)
from pmwd.ray_mesh import precompute_mesh

def load_config(config_name):
    with open('/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/fin_test/config_run.json', 'r') as file:
        configs = json.load(file)
    return configs.get(config_name)

def run(config, config_name, sample_id):
    # Unpack configuration parameters
    ptcl_spacing = config['ptcl_spacing']
    ptcl_grid_shape = tuple(config['ptcl_grid_shape'])
    mesh_shape = config.get('mesh_shape', 1)  # Default value if not specified
    rs = config['rs']
    ray_shape = tuple(config['ray_shape'])
    so = config['so']
    t_step = config['t_step']
    z_rt_end = config['z_rt_end']
    iota = config['iota']
    padding = config['padding']

    save_folder = f'/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/fin_test/data/{config_name}'
    save_path = os.path.join(save_folder, f'sample_{sample_id}.npz')
    os.makedirs(save_folder, exist_ok=True)        

    ray_grid_shape = ray_shape
    ray_spacing = rs / 60 * jnp.pi / 180

    h = 0.6774
    Omega_m = 0.3089
    Omega_b = 0.0486
    Omega_c = Omega_m - Omega_b
    n_s = 0.9667
    sigma8 = 0.8159

    conf_params = {
        'ptcl_spacing': ptcl_spacing,
        'ptcl_grid_shape': ptcl_grid_shape,
        'ray_spacing': ray_spacing,
        'ray_grid_shape': ray_grid_shape,
        'mesh_shape': mesh_shape,
        'ray_mesh_shape_default': 2,
        'z_rt_end': z_rt_end,
        'a_nbody_rt_num': t_step,
        'ray_mesh_iota': iota,
        'ray_mesh_p_x': padding,
        'ray_mesh_p_y': padding,
    }
    if so:
        so_params, _, so_nodes = load_soparams('/hildafs/projects/phy230056p/junzhez/pmwd/pmwd/sto/sonn_params.pickle')
        conf_params.update({
            'so_type': 'NN',
            'so_nodes': so_nodes,
            'soft_i': 'soft_v2',
            'softening_length': 0.02
        })
    else:
        so_params = None

    conf = Configuration(**conf_params)
    cosmo = Cosmology.from_sigma8(conf, sigma8=sigma8, h=h, Omega_b=Omega_b, Omega_m=Omega_m, n_s=n_s, so_params=so_params)
    cosmo = boltzmann(cosmo, conf)
    ray_mesh_params = precompute_mesh(conf, cosmo)  
     
    if sample_id == 0:
        plot_name = os.path.join(save_folder, f'diagram.png')
        lens_plane_diagram(cosmo, conf, name=plot_name)

    print(f'Processing sample {sample_id}')

    modes = white_noise(sample_id, conf)
    modes = linear_modes(modes, cosmo, conf)

    ray = Rays.gen_grid(conf)
    obsvbl_ray = jnp.zeros((4, conf.ray_num), dtype=jnp.float64)

    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    ptcl, ray, obsvbl, obsvbl_ray = nbody_ray(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf, ray_mesh_params)
    kappa, gamma1, gamma2, omega = obsvbl_ray


    np.savez(
        save_path,
        sample_id=sample_id,
        kappa=kappa,
        gamma1=gamma1,
        gamma2=gamma2,
        omega=omega,
        ray_pos=ray.pos_ip(),
    )

def main(config_name):
    config = load_config(config_name)
    if not config:
        print(f"Configuration '{config_name}' not found.")
        return

    for sample_id in range(50):
        run(config, config_name, sample_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help="The name of the configuration to run.")
    args = parser.parse_args()

    print(f"Running configuration: {args.config_name}")
    main(args.config_name)
    print('Done')
