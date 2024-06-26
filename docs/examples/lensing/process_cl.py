import os, sys
import jax
import jax.numpy as jnp
import numpy as np


def run_theory(z_rt_end):
    import pyccl as ccl

    h = 0.6774
    Omega_m = 0.3089
    Omega_b = 0.0486
    Omega_c = Omega_m - Omega_b
    n_s = 0.9667
    sigma8 = 0.8159

    cosmo_pyccl = ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        n_s=n_s,
        sigma8=sigma8,
        transfer_function="bbks",
    )

    z_si = z_rt_end - 0.01
    z_sf = z_rt_end + 0.01
    z = jnp.linspace(0, z_sf * 1.2, 256)
    nz = jnp.ones_like(z)
    nz = jnp.where((z > z_si) & (z < z_sf), nz, 0)
    wl = ccl.WeakLensingTracer(cosmo_pyccl, dndz=(z, nz))
    l = np.arange(20, 3000)
    cl_ccl = ccl.angular_cl(cosmo_pyccl, wl, wl, l) * l * (l + 1) / (2 * np.pi)
    return l, cl_ccl


############################################
# Load data
############################################
def get_save_folder(config):
    """
    Generate a file path based on the configuration settings.

    Parameters:
    config (dict): Configuration dictionary containing simulation parameters.

    Returns:
    str: The complete file path where the results should be saved.
    """
    # Unpack configuration parameters
    ptcl_spacing = config.get('ptcl_spacing')
    ptcl_grid_shape = tuple(config.get('ptcl_grid_shape'))
    mesh_shape = config.get('mesh_shape', 1)  # Provide a default value if not set
    rs = config.get('rs')
    ray_shape = tuple(config.get('ray_shape'))
    so = config.get('so')
    t_step = config.get('t_step')
    z_rt_end = config.get('z_rt_end')
    iota = config.get('iota')
    padding = config.get('padding')

    # Construct the directory path
    save_folder = os.path.join(
        '/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/new/data_cl',
        f'ptcl_{ptcl_spacing}_{ptcl_grid_shape}_{mesh_shape}_ray_{rs}_{ray_shape}_so_{so}_t_{t_step}_i_{iota}_p_{padding}'
    )
    return save_folder


def get_cls(config, seeds=[0], l_interp=None):
    from grid_analysis import GridAnalysis
    # set up grid
    ray_spacing = config.get('rs') / 60 * jnp.pi / 180
    ray_grid_shape = config.get('ray_shape')
    xlim = np.array([-ray_spacing*ray_grid_shape[0]/2, ray_spacing*ray_grid_shape[0]/2])
    ylim = np.array([-ray_spacing*ray_grid_shape[1]/2, ray_spacing*ray_grid_shape[1]/2])
    xlim *= 180/np.pi
    ylim *= 180/np.pi
    grid = GridAnalysis(xlim=xlim, ylim=ylim, xn=ray_grid_shape[0], yn=ray_grid_shape[1], nlbin=300)

    save_folder = get_save_folder(config)
    print('from folder',save_folder)
    
    l = grid.pse_lvals
    cl_kappa = []
    cl_gamma1 = []
    cl_gamma2 = []
    cl_omega = []
    for seed in seeds:
        try:
            path = os.path.join(save_folder, f'seed_{seed}.npz')
            data = np.load(path)
            kappa = data['kappa'].reshape(ray_grid_shape)
            gamma1 = data['gamma1'].reshape(ray_grid_shape)
            gamma2 = data['gamma2'].reshape(ray_grid_shape)
            omega = data['omega'].reshape(ray_grid_shape)
            
            cl_kappa.append(grid.PSE_XPS(jnp.array([kappa]))[0,0] * l * (l + 1) / (2 * np.pi))
            cl_gamma1.append(grid.PSE_XPS(jnp.array([gamma1]))[0,0] * l * (l + 1) / (2 * np.pi))
            cl_gamma2.append(grid.PSE_XPS(jnp.array([gamma2]))[0,0] * l * (l + 1) / (2 * np.pi))
            cl_omega.append(grid.PSE_XPS(jnp.array([omega]))[0,0] * l * (l + 1) / (2 * np.pi))
        except:
            print(f'seed {seed} does not exist')
            continue
    cl_kappa = np.array(cl_kappa)
    cl_gamma1 = np.array(cl_gamma1)
    cl_gamma2 = np.array(cl_gamma2)
    cl_omega = np.array(cl_omega)
    n_samples = cl_kappa.shape[0]
    print(np.asarray(cl_kappa).shape)
    
    # mean, -1 Std, +1 Std
    # cl_kappa = [np.mean(cl_kappa, axis=0), np.percentile(cl_kappa, 16, axis=0), np.percentile(cl_kappa, 84, axis=0)]
    # cl_gamma1 = [np.mean(cl_gamma1, axis=0), np.percentile(cl_gamma1, 16, axis=0), np.percentile(cl_gamma1, 84, axis=0)]
    # cl_gamma2 = [np.mean(cl_gamma2, axis=0), np.percentile(cl_gamma2, 16, axis=0), np.percentile(cl_gamma2, 84, axis=0)]
    # cl_omega = [np.mean(cl_omega, axis=0), np.percentile(cl_omega, 16, axis=0), np.percentile(cl_omega, 84, axis=0)]
    # # calculate standard error of the mean (std/sqrt(n))
    cl_kappa_mean = np.mean(cl_kappa, axis=0)
    cl_gamma1_mean = np.mean(cl_gamma1, axis=0)
    cl_gamma2_mean = np.mean(cl_gamma2, axis=0)
    cl_omega_mean = np.mean(cl_omega, axis=0)
    
    cl_kappa_se = np.std(cl_kappa, axis=0) / np.sqrt(n_samples)
    cl_gamma1_se = np.std(cl_gamma1, axis=0) / np.sqrt(n_samples)
    cl_gamma2_se = np.std(cl_gamma2, axis=0) / np.sqrt(n_samples)
    cl_omega_se = np.std(cl_omega, axis=0) / np.sqrt(n_samples)
    
    cl_kappa_std = np.std(cl_kappa, axis=0)
    cl_gamma1_std = np.std(cl_gamma1, axis=0)
    cl_gamma2_std = np.std(cl_gamma2, axis=0)
    cl_omega_std = np.std(cl_omega, axis=0)
    
    cl_kappa = [cl_kappa_mean, cl_kappa_mean - cl_kappa_se, cl_kappa_mean + cl_kappa_se]
    cl_gamma1 = [cl_gamma1_mean, cl_gamma1_mean - cl_gamma1_se, cl_gamma1_mean + cl_gamma1_se]
    cl_gamma2 = [cl_gamma2_mean, cl_gamma2_mean - cl_gamma2_se, cl_gamma2_mean + cl_gamma2_se]
    cl_omega = [cl_omega_mean, cl_omega_mean - cl_omega_se, cl_omega_mean + cl_omega_se]
    
    # cl_kappa = [cl_kappa_mean, cl_kappa_mean - cl_kappa_std, cl_kappa_mean + cl_kappa_std]
    # cl_gamma1 = [cl_gamma1_mean, cl_gamma1_mean - cl_gamma1_std, cl_gamma1_mean + cl_gamma1_std]
    # cl_gamma2 = [cl_gamma2_mean, cl_gamma2_mean - cl_gamma2_std, cl_gamma2_mean + cl_gamma2_std]
    # cl_omega = [cl_omega_mean, cl_omega_mean - cl_omega_std, cl_omega_mean + cl_omega_std]
    if l_interp is not None:
        # first if cl[0] is nan, drop the (l, cl[0], cl[1], cl[2]) pairs
        # cl_kappa = [np.where(np.isnan(cl_kappa[0]), np.nan, cl_kappa[0]), np.where(np.isnan(cl_kappa[0]), np.nan, cl_kappa[1]), np.where(np.isnan(cl_kappa[0]), np.nan, cl_kappa[2])]
        # cl_gamma1 = [np.where(np.isnan(cl_gamma1[0]), np.nan, cl_gamma1[0]), np.where(np.isnan(cl_gamma1[0]), np.nan, cl_gamma1[1]), np.where(np.isnan(cl_gamma1[0]), np.nan, cl_gamma1[2])]
        # cl_gamma2 = [np.where(np.isnan(cl_gamma2[0]), np.nan, cl_gamma2[0]), np.where(np.isnan(cl_gamma2[0]), np.nan, cl_gamma2[1]), np.where(np.isnan(cl_gamma2[0]), np.nan, cl_gamma2[2])]
        # cl_omega = [np.where(np.isnan(cl_omega[0]), np.nan, cl_omega[0]), np.where(np.isnan(cl_omega[0]), np.nan, cl_omega[1]), np.where(np.isnan(cl_omega[0]), np.nan, cl_omega[2])]
        cl_kappa = [np.interp(l_interp, l, cl_kappa[0]), np.interp(l_interp, l, cl_kappa[1]), np.interp(l_interp, l, cl_kappa[2])]
        cl_gamma1 = [np.interp(l_interp, l, cl_gamma1[0]), np.interp(l_interp, l, cl_gamma1[1]), np.interp(l_interp, l, cl_gamma1[2])]
        cl_gamma2 = [np.interp(l_interp, l, cl_gamma2[0]), np.interp(l_interp, l, cl_gamma2[1]), np.interp(l_interp, l, cl_gamma2[2])]
        cl_omega = [np.interp(l_interp, l, cl_omega[0]), np.interp(l_interp, l, cl_omega[1]), np.interp(l_interp, l, cl_omega[2])]
        l = l_interp
        
    return l, (cl_kappa, cl_gamma1, cl_gamma2, cl_omega)


def get_map(config, seed):
    save_folder = get_save_folder(config)
    path = os.path.join(save_folder, f'seed_{seed}.npz')
    data = np.load(path)

    ray_grid_shape = config.get('ray_shape')
    kappa = data['kappa'].reshape(ray_grid_shape)
    gamma1 = data['gamma1'].reshape(ray_grid_shape)
    gamma2 = data['gamma2'].reshape(ray_grid_shape)
    omega = data['omega'].reshape(ray_grid_shape)
    
    # grid
    ray_spacing = config.get('rs') / 60 * jnp.pi / 180
    xlim = np.array([-ray_spacing*ray_grid_shape[0]/2, ray_spacing*ray_grid_shape[0]/2])
    ylim = np.array([-ray_spacing*ray_grid_shape[1]/2, ray_spacing*ray_grid_shape[1]/2])
    xlim *= 180/np.pi * 60
    ylim *= 180/np.pi * 60
    
    x,y = np.meshgrid(np.linspace(xlim[0], xlim[1], ray_grid_shape[0]), np.linspace(ylim[0], ylim[1], ray_grid_shape[1]))
    return kappa, gamma1, gamma2, omega, x, y

############################################
# Plotting
############################################

def plot_error(ax, l_HRT, cl_HRT, l_theory, cl_theory, **kwargs):
    """
    Plot error between HRT and theory
    """
    # cast as np array
    l_HRT = jnp.array(l_HRT)
    cl_HRT = jnp.array(cl_HRT)
    l_theory = jnp.array(l_theory)
    cl_theory = jnp.array(cl_theory)

    # interpolate to l
    cl_theory_interp = np.interp(l_HRT, l_theory, cl_theory)
    error = (cl_HRT / cl_theory_interp) - 1

    ax.fill_between(l_HRT, error[1], error[2], **kwargs)
    ax.plot(l_HRT, error[0])
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$(C_{\ell}^{\rm{HRT}} - C_{\ell}^{\rm{theory}}) / C_{\ell}^{\rm{theory}}$')
    return


def plot(configs, varname, vartex, z_rt_end, seeds=list(range(10)), l=None, plot_xlim=(150,1500), relerr='tng'):
    from matplotlib import pyplot as plt
    from grid_analysis import GridAnalysis
    from load_kTNG import filename,load
        
    plt.close()
    f, axs = plt.subplot_mosaic(
        [
        ["cl",],
        ["error",],
    ],
        figsize=(8, 6),
        gridspec_kw={
            "height_ratios": [1.0, 0.5],
            "width_ratios": [1.0],
        },
        sharex=True,
    )
    # theory
    if relerr == 'theory':
        l_theory, cl_theory = run_theory(z_rt_end)
        axs["cl"].plot(l_theory, cl_theory, label='theory', color='k', ls='-')
    
    # TNG 
    xlim,ylim = [-2.5,2.5], [-2.5,2.5]
    xn, yn = 1024, 1024
    grid_kTNG = GridAnalysis(xlim=xlim, ylim=ylim, xn=xn, yn=yn, nlbin=400)
    l_kTNG = grid_kTNG.pse_lvals
    cl_kTNG = np.zeros((100, l_kTNG.size))
    for index in range(1,100):
        fname = filename(index)
        kappa, _ = load(fname)
        cl_tmp = grid_kTNG.PSE_XPS(jnp.array([kappa]))[0,0] * l_kTNG * (l_kTNG + 1) / (2 * np.pi)
        cl_kTNG[index] = cl_tmp
    axs["cl"].plot(l_kTNG, np.mean(cl_kTNG, axis=0), label='$\kappa$TNG-Dark', color='k', ls='--')

    # pmwd
    for index, config in enumerate(configs):
        var = config.get(varname)
        label = f"{vartex} = {var}"
        if varname =='so':
            label = 'with SO' if var else 'without SO'
                
        l, cl = get_cls(config, seeds=seeds, l_interp=l)
        
        kwargs_fill = {"alpha": 0.3, 'facecolor': f"C{index}", 'label': f'HRT, {label}','edgecolor':'none'}
        
        axs["cl"].fill_between(l, cl[0][1], cl[0][2], **kwargs_fill)
        axs["cl"].plot(l, cl[0][0], color=f"C{index}")
        if relerr == 'tng':
            plot_error(axs["error"], l, cl[0], l_kTNG, np.mean(cl_kTNG, axis=0), **kwargs_fill)
        else:            
            plot_error(axs["error"], l, cl[0], l_theory, cl_theory, **kwargs_fill)
    
    if relerr == 'theory':
        axs["error"].axhline(0, color="black", linestyle="-")
        cl_theory_interp = np.interp(l_kTNG, l_theory, cl_theory)
        error = (np.mean(cl_kTNG, axis=0) / cl_theory_interp) - 1
        axs["error"].plot(l_kTNG, error, color='k', ls='--')
    else:
        axs["error"].axhline(0, color="black", linestyle="--")
    axs["cl"].set_xscale("log")
    axs["cl"].set_yscale("log")
    axs["cl"].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    axs["cl"].set_ylim(1e-6, 1e-3)
    axs["cl"].tick_params(axis="x", direction="in", which="both")

    axs['error'].axhline(0, color="black", linestyle="-")
    axs['error'].set_xscale("log")
    axs['error'].set_yscale("linear")
    axs['error'].set_xlabel(r"$\ell$")
    axs['error'].set_ylabel("relative error \n in $C_\ell$")
    
    axs['error'].set_xlim(plot_xlim)
    axs['error'].set_ylim(-1.2, 1.2)
    return f, axs

def plot_map(ax, config, seed, **kwargs):
    kappa, gamma1, gamma2, omega, x, y = get_map(config, seed)
    # c = ax.imshow(kappa, **kwargs)
    c = ax.pcolormesh(x, y, kappa, **kwargs)
    ax.set_aspect('equal')
    return c

def plot_map_pdf(ax, config, seeds, need_gaussian=True, **kwargs):
    bin_lims = (-0.04, 0.06)
    kappa_list = []
    for seed in seeds:
        kappa, gamma1, gamma2, omega, x, y = get_map(config, seed)
        kappa_list.append(kappa)
    kappa_list = np.array(kappa_list)
    
    if need_gaussian:
        # plot gaussian of the same variance
        from scipy.stats import norm
        mu = 0
        sigma = np.std(kappa)
        x = np.linspace(bin_lims[0], bin_lims[1], 100)
        ax.plot(x, norm.pdf(x, mu, sigma), 'k--', label='Gaussian')
        
    ax.hist(kappa_list.flatten(), bins=100, range=bin_lims, density=True, histtype='step', **kwargs)
    return
