import os
import sys
import json
import argparse

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
sys.path.insert(0, '/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/new')

import jax
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

def load_config(config_name):
    with open('/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/new/config_run.json', 'r') as file:
        configs = json.load(file)
    return configs.get(config_name)

def run(config, seed):
    # Unpack configuration parameters
    ptcl_spacing = config.get('ptcl_spacing')
    ptcl_grid_shape = tuple(config.get('ptcl_grid_shape'))
    mesh_shape = config.get('mesh_shape', 1)  # Default value example
    rs = config.get('rs')
    ray_shape = tuple(config.get('ray_shape'))
    so = config.get('so')
    t_step = config.get('t_step')
    z_rt_end = config.get('z_rt_end')
    iota = config.get('iota')
    padding = config.get('padding')

    save_folder = f'/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/new/data_cl'
    save_folder = os.path.join(save_folder, f'ptcl_{ptcl_spacing}_{ptcl_grid_shape}_{mesh_shape}_ray_{rs}_{ray_shape}_so_{so}_t_{t_step}_i_{iota}_p_{padding}')
    save_path = os.path.join(save_folder, f'seed_{seed}.npz')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ray_grid_shape = ray_shape
    ray_spacing = rs / 60 * jnp.pi / 180

    h = 0.6774
    Omega_m = 0.3089
    Omega_b = 0.0486
    Omega_c = Omega_m - Omega_b
    n_s = 0.9667
    sigma8 = 0.8159

    if so:
        so_params, n_input, so_nodes = load_soparams('/hildafs/projects/phy230056p/junzhez/pmwd/pmwd/sto/sonn_params.pickle')
        conf = Configuration(
            ptcl_spacing,
            ptcl_grid_shape,
            ray_spacing,
            ray_grid_shape,
            mesh_shape=mesh_shape,
            ray_mesh_shape_default=2,
            z_rt_end=z_rt_end,
            a_nbody_rt_num=t_step,
            ray_mesh_iota=iota,
            ray_mesh_p_x=padding,
            ray_mesh_p_y=padding,
            so_type='NN', so_nodes=so_nodes, soft_i='soft_v2', softening_length=0.02
        )
        cosmo = Cosmology.from_sigma8(conf, sigma8=sigma8, h=h, Omega_b=Omega_b, Omega_m=Omega_m, n_s=n_s, so_params=so_params)
    else:
        conf = Configuration(
            ptcl_spacing,
            ptcl_grid_shape,
            ray_spacing,
            ray_grid_shape,
            mesh_shape=mesh_shape,
            ray_mesh_shape_default=2,
            z_rt_end=z_rt_end,
            a_nbody_rt_num=t_step,
            ray_mesh_iota=iota,
            ray_mesh_p_x=padding,
            ray_mesh_p_y=padding,
        )
        cosmo = Cosmology.from_sigma8(conf, sigma8=sigma8, h=h, Omega_b=Omega_b, Omega_m=Omega_m, n_s=n_s)
    cosmo = boltzmann(cosmo, conf)

    if seed == 0:
        plot_name = os.path.join(f'/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/new/data_cl', f'ptcl_{ptcl_spacing}_{ptcl_grid_shape}_{mesh_shape}_ray_{rs}_{ray_shape}_so_{so}_t_{t_step}_i_{iota}_p_{padding}.png')
        lens_plane_diagram(cosmo, conf, name=plot_name)

    print('seed', seed)

    modes = white_noise(seed, conf)
    modes = linear_modes(modes, cosmo, conf)

    ray = Rays.gen_grid(conf)
    obsvbl_ray = jnp.zeros((4, conf.ray_num), dtype=jnp.float64)

    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    ptcl, ray, obsvbl, obsvbl_ray = nbody_ray(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf)
    kappa, gamma1, gamma2, omega = obsvbl_ray

    np.savez(
        save_path,
        seed=seed,
        kappa=kappa,
        gamma1=gamma1,
        gamma2=gamma2,
        omega=omega,
        ray_pos=ray.pos_ip(),
    )
    return

def main(config_name):
    config = load_config(config_name)
    if not config:
        print(f"Configuration '{config_name}' not found.")
        return

    seeds = list(range(0, 50))
    for seed in seeds:
        run(config, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help="The name of the configuration to run.")
    args = parser.parse_args()

    print(args.config_name)
    main(args.config_name)
    print('Done')
