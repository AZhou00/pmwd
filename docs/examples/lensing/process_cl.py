import os, sys
import jax
import jax.numpy as jnp
import numpy as np
from grid_analysis import GridAnalysis

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
def get_data_path(config_name, sample):
    """Generate a file path for a specific sample."""
    return f'/hildafs/projects/phy230056p/junzhez/pmwd_test/pmwd_raytracing/fin_test/data/{config_name}/sample_{sample}.npz'

def get_cls(config, config_name, samples=[0], l_interp=None):
    print('from folder', os.path.dirname(get_data_path(config_name, samples[0])))
    
    # Set up grid parameters
    ray_spacing = config.get('rs') / 60 * jnp.pi / 180
    ray_grid_shape = config.get('ray_shape')
    xlim = np.array([-ray_spacing*ray_grid_shape[0]/2, ray_spacing*ray_grid_shape[0]/2]) * 180/np.pi
    ylim = np.array([-ray_spacing*ray_grid_shape[1]/2, ray_spacing*ray_grid_shape[1]/2]) * 180/np.pi
    grid = GridAnalysis(xlim=xlim, ylim=ylim, xn=ray_grid_shape[0], yn=ray_grid_shape[1], nlbin=300)
    
    l = grid.pse_lvals
    cls = {field: [] for field in ['kappa', 'gamma1', 'gamma2', 'omega']}

    # Load and process data for each sample
    for sample in samples:
        try:
            data = np.load(get_data_path(config_name, sample))
            for field in cls.keys():
                field_data = data[field].reshape(ray_grid_shape)
                cls[field].append(grid.PSE_XPS(jnp.array([field_data]))[0,0] * l * (l + 1) / (2 * np.pi))
        except:
            print(f'sample {sample} does not exist')
            continue

    # Calculate mean and standard error
    n_samples = len(cls['kappa'])
    results = {}
    for field in cls.keys():
        field_data = np.array(cls[field])
        mean = np.mean(field_data, axis=0)
        se = np.std(field_data, axis=0) / np.sqrt(n_samples)
        results[field] = [mean, mean - se, mean + se]

    # Interpolate if necessary
    if l_interp is not None:
        for field in results.keys():
            results[field] = [np.interp(l_interp, l, cl) for cl in results[field]]
        l = l_interp
        
    return l, (results['kappa'], results['gamma1'], results['gamma2'], results['omega'])

def get_map(config, config_name, sample):
    # Load data for a specific sample
    data = np.load(get_data_path(config_name, sample))

    # Reshape fields into 2D maps
    ray_grid_shape = config.get('ray_shape')
    fields = ['kappa', 'gamma1', 'gamma2', 'omega']
    maps = [data[field].reshape(ray_grid_shape) for field in fields]
    
    # Calculate x and y coordinates for the map
    ray_spacing = config.get('rs') / 60 * jnp.pi / 180
    xlim = np.array([-ray_spacing*ray_grid_shape[0]/2, ray_spacing*ray_grid_shape[0]/2]) * 180/np.pi * 60
    ylim = np.array([-ray_spacing*ray_grid_shape[1]/2, ray_spacing*ray_grid_shape[1]/2]) * 180/np.pi * 60
    
    x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], ray_grid_shape[0]), 
                       np.linspace(ylim[0], ylim[1], ray_grid_shape[1]))
    return *maps, x, y

def plot_error(ax, l_HRT, cl_HRT, l_theory, cl_theory, **kwargs):
    l_HRT, cl_HRT = map(jnp.array, (l_HRT, cl_HRT))
    l_theory, cl_theory = map(jnp.array, (l_theory, cl_theory))

    cl_theory_interp = np.interp(l_HRT, l_theory, cl_theory)
    error = (cl_HRT / cl_theory_interp) - 1

    ax.fill_between(l_HRT, error[1], error[2], **kwargs)
    ax.plot(l_HRT, error[0])
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$(C_{\ell}^{\rm{HRT}} - C_{\ell}^{\rm{theory}}) / C_{\ell}^{\rm{theory}}$')

def plot(configs, config_names, varname, vartex, z_rt_end, samples=range(10), l=None, plot_xlim=(150,1500), relerr='tng', plot_ktng=True):
    from matplotlib import pyplot as plt
    from load_kTNG import filename, load
    
    # setup plot
    plt.close()
    f, axs = plt.subplot_mosaic([["cl"], ["error"]], figsize=(8, 6),
                                gridspec_kw={"height_ratios": [1.0, 0.5]},
                                sharex=True)
    
    # plot theory
    l_theory, cl_theory = run_theory(z_rt_end)
    axs["cl"].plot(l_theory, cl_theory, label='theory', color='k', ls='-')
    
    if plot_ktng:
        grid_kTNG = GridAnalysis(xlim=[-2.5,2.5], ylim=[-2.5,2.5], xn=1024, yn=1024, nlbin=400)
        l_kTNG = grid_kTNG.pse_lvals
        cl_kTNG = np.array([grid_kTNG.PSE_XPS(jnp.array([load(filename(i))[0]]))[0,0] * l_kTNG * (l_kTNG + 1) / (2 * np.pi)
                            for i in range(1, 100)])
        axs["cl"].plot(l_kTNG, np.mean(cl_kTNG, axis=0), label='$\kappa$TNG-Dark', color='k', ls='--')

    for index, (config, config_name) in enumerate(zip(configs,config_names)):
        # plot settings
        var = config.get(varname)
        label = f"{vartex} = {var}" if varname != 'so' else ('with SO' if var else 'without SO')
        kwargs_fill = {"alpha": 0.3, 'facecolor': f"C{index}", 'label': f'HRT, {label}', 'edgecolor':'none'}
        
        # cl plot
        l, cl = get_cls(config, config_name, samples=samples, l_interp=l)
        axs["cl"].fill_between(l, cl[0][1], cl[0][2], **kwargs_fill)
        axs["cl"].plot(l, cl[0][0], color=f"C{index}")
        # error plot
        plot_error(axs["error"], l, cl[0], l_kTNG if relerr == 'tng' else l_theory, 
                   np.mean(cl_kTNG, axis=0) if relerr == 'tng' else cl_theory, **kwargs_fill)

    # error plot for tng
    if relerr == 'theory':
        if plot_ktng:   
            cl_theory_interp = np.interp(l_kTNG, l_theory, cl_theory)
            error = (np.mean(cl_kTNG, axis=0) / cl_theory_interp) - 1
            axs["error"].plot(l_kTNG, error, color='k', ls='--')
    # error plot grid line
    axs["error"].axhline(0, color="black", linestyle="-" if relerr == 'theory' else "--")
    
    for ax in axs.values():
        ax.set_xscale("log")
    axs["cl"].set_yscale("log")
    axs["cl"].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    axs["cl"].tick_params(axis="x", direction="in", which="both")

    axs['error'].set_xlabel(r"$\ell$")
    axs['error'].set_ylabel("relative error \n in $C_\ell$")
    axs['error'].set_xlim(plot_xlim)
    axs['error'].set_ylim(-1.2, 1.2)

    return f, axs

def plot_map(ax, config, sample, **kwargs):
    kappa, _, _, _, x, y = get_map(config, sample)
    c = ax.pcolormesh(x, y, kappa, **kwargs)
    ax.set_aspect('equal')
    return c

def plot_map_pdf(ax, config, samples, need_gaussian=True, **kwargs):
    bin_lims = (-0.04, 0.06)
    kappa_list = [get_map(config, sample)[0] for sample in samples]
    kappa_list = np.array(kappa_list)
    
    if need_gaussian:
        from scipy.stats import norm
        sigma = np.std(kappa_list)
        x = np.linspace(bin_lims[0], bin_lims[1], 100)
        ax.plot(x, norm.pdf(x, 0, sigma), 'k--', label='Gaussian')
        
    ax.hist(kappa_list.flatten(), bins=100, range=bin_lims, density=True, histtype='step', **kwargs)
