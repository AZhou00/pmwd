import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
from pmwd.boltzmann import chi_a
import numpy as np
# https://github.com/lenstronomy/lenstronomy-tutorials/blob/main/Notebooks/GettingStarted/starting_guide.ipynb
# https://colab.research.google.com/github/sibirrer/strong_lensing_lectures/blob/main/Lectures/lensing_basics_I.ipynb

def point_mass_deflection(theta0, chi_l, chi_s, M, conf, unit = 'arcmin'):
    """
    Deflection angle for point mass for flat universe.
    
    Parameters
    ----------
    theta0 : float
        Initial position of the ray, shape = [ray_num, 2], in radian
    chi_l : float
        Comoving distance to lens, shape = () or [ray_num], in Mpc
    chi_s : float
        Comoving distance to source, shape = () or [ray_num], in Mpc
    M : float
        Mass of the lens, shape = () or [ray_num], in M
        
    Returns
    -------
    theta_s : float
        Final position of the ray, shape = [ray_num, 2], in radians.
    """
    # jnp.abs(theta0,axis=-1) # has shape [ray_num]/
    chi_ls = chi_s - chi_l
    theta0_mag = jnp.linalg.norm(theta0,axis=1)[:,None]
    alpha = 4 * conf.G * M / (conf.c ** 2)
    alpha *= (chi_ls / chi_s / chi_l)  * (theta0 / theta0_mag**2)
    theta_s = theta0 - alpha
    
    if unit == 'arcmin':
        theta_s *= 180*60/jnp.pi
        alpha *= 180*60/jnp.pi
        
    return theta_s, - alpha

def visual_ray_point_mass(theta0, theta_prev, theta_s, chi_l, chi_s_prev, chi_s, M, conf, ptcl):
    """
    Visualize the ray positions on the image plane.
    Assume source at chi_s and theta_s is the inferred true angular position.
    """
    f, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    
    # point mass position
    ptcl_coord2D = ptcl.pos()[0,0:2]
    ptcl_coord2D -= jnp.array(conf.box_size[0:2])/2
    ptcl_coord2D /= ptcl.pos()[0,2]

    # Einstein radius
    theta_E = jnp.sqrt((chi_s-chi_l) / (chi_s) / chi_l * 4 * conf.G * M / conf.c**2) * 3437.75

    # theory point mass deflection angle, input in radians
    theta_theory_prev, _ = point_mass_deflection(theta0, chi_l, chi_s_prev, M, conf, unit='arcmin')
    theta_theory, _ = point_mass_deflection(theta0, chi_l, chi_s, M, conf, unit='arcmin')
    
    # unit scaling
    theta0 = theta0 * 3437.75
    theta_prev = theta_prev * 3437.75
    theta_s = theta_s * 3437.75
    
    scale = 1.5
    mask = jnp.linalg.norm(theta_theory, axis=-1) > theta_E * scale 
    mask = mask & (jnp.linalg.norm(theta_s, axis=-1) > theta_E * scale)

    theta0 = theta0[mask]
    theta_prev = theta_prev[mask]
    theta_s = theta_s[mask]
    theta_theory_prev = theta_theory_prev[mask]
    theta_theory = theta_theory[mask]

    axs[0].scatter(ptcl_coord2D[0], ptcl_coord2D[1], s=16, color="k", label="ray-tracing")
    axs[0].add_patch(
        plt.Circle((0, 0), theta_E, fill=False, color="k", label=r"$\theta_E$")
    )
    axs[0].scatter(
        theta_s[..., 0],
        theta_s[..., 1],
        s=2,
        color="r",
        label="ray-tracing",
    )
    axs[0].scatter(
        theta_theory[..., 0],
        theta_theory[..., 1],
        s=2,
        color="b",
        label="theory",
    )
    limit = 75
    axs[0].set_xlim(-limit, limit)
    axs[0].set_ylim(-limit, limit)
    axs[0].set_aspect("equal")
    axs[0].grid()
    # top legend
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)

    # residual plot, deflection angle
    r = jnp.linalg.norm(theta_theory, axis=-1)
    dtheta_rt = theta_s - theta0
    dtheta_theory = theta_theory - theta0
    residual = jnp.linalg.norm(theta_theory - theta_s, axis=-1)
    # print(theta_theory[0:5], theta_s[0:5])
    axs[1].scatter(r, jnp.linalg.norm(dtheta_rt, axis=-1), s=1, color="r", label="ray-traced")
    axs[1].scatter(r, jnp.linalg.norm(dtheta_theory, axis=-1), s=1, color="b", label="theory")
    axs[1].scatter(r, residual, s=1, color="k", label=r"$|\theta_\mathrm{t,n} - \theta_\mathrm{rt,n}|$")
    axs[1].set_ylabel(r"$|\theta|$ [arcmin]",fontsize=12)
    axs[1].set_xlabel(r"$|\theta_\mathrm{t,n} - \theta_0|$ [arcmin]",fontsize=12)
    axs[1].legend()
    plt.show()
    
    
    # dtheta_rt_norm = jnp.linalg.norm(dtheta_rt, axis=-1)
    # dtheta_theory_norm = jnp.linalg.norm(dtheta_theory, axis=-1)
    # residual = jnp.linalg.norm(dtheta_rt - dtheta_theory, axis=-1)
    # axs[1].scatter(r, dtheta_rt_norm, s=1, color="r", label="ray-traced")
    # axs[1].scatter(r, dtheta_theory_norm, s=1, color="b", label="theory")
    # axs[1].scatter(r, residual, s=1, color="k", label="residual")
    # axs[1].set_ylabel("deflection angle (arcmin)")
    # axs[1].set_xlabel("radius from LOS (arcmin)")
    # axs[1].set_yscale("log")
    # axs[1].legend()
    # axs[1].set_ylim(1e-3*1e1,1e-1*1e1)
    dtheta_rt_norm = jnp.linalg.norm(dtheta_rt, axis=-1)
        
    plt.show()
    return

def lens_plane_diagram(cosmo, conf, chi_l=None):
    from matplotlib.patches import Arc

    f, ax = plt.subplots(1,1,figsize=(12,12))
    
    xmin = 0
    xmax = conf.box_size[2]
    ymin = -conf.box_size[0]/2
    ymax = conf.box_size[0]/2
    
    chi_s = chi_a(conf.a_nbody_ray[-1], cosmo, conf)
    
    # box
    ax.plot([xmin,xmin,xmax,xmax,xmin], [ymin,ymax,ymax,ymin,ymin], 'k')
    
    # light cones arc
    theta_min = np.arctan2(-conf.box_size[0]/2, chi_s)
    theta_max = np.arctan2(conf.box_size[0]/2, chi_s)
    
    ip_fov = conf.ray_spacing * conf.ray_grid_shape[0] # in radians
    theta_min = -ip_fov / 2
    theta_max = ip_fov / 2
    
    ax.add_patch(Arc((0,0), 2*chi_s, 2*chi_s, angle=0, theta1=theta_min*180/np.pi, theta2=theta_max*180/np.pi, color='r'))
    # light cones lines
    x1 = chi_s * np.cos(theta_min)
    x2 = chi_s * np.cos(theta_max)
    y1 = chi_s * np.sin(theta_min)
    y2 = chi_s * np.sin(theta_max)
    ax.add_line(plt.Line2D([0, x1], [0, y1], color='r'))
    ax.add_line(plt.Line2D([0, x1], [0, y2], color='r'))
    
    a_lens_planes = np.concatenate([conf.a_nbody_ray, (conf.a_nbody_ray[1:] + conf.a_nbody_ray[:-1]) / 2])
    a_lens_planes = np.sort(a_lens_planes)
    chi_lens = chi_a(a_lens_planes, cosmo, conf)
    print('lenses at chi =', chi_lens)
    print('nbody steps at chi =', chi_a(conf.a_nbody_ray, cosmo, conf))
    for chi in chi_lens[1:-1]:
        ax.add_line(plt.Line2D([chi, chi], [theta_min*chi, theta_max*chi], color='b', linestyle='--'))
    
    if chi_l is not None:
        # plot lensing mass
        ax.plot(chi_l, 0, 'kx')
    
    ax.set_aspect('equal')
    plt.show()
