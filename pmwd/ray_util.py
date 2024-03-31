import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

# https://github.com/lenstronomy/lenstronomy-tutorials/blob/main/Notebooks/GettingStarted/starting_guide.ipynb
# https://colab.research.google.com/github/sibirrer/strong_lensing_lectures/blob/main/Lectures/lensing_basics_I.ipynb


c_SI = 299792458  # m/s
G_SI = 6.67430e-11  # m^3/kg/s^2
M_sun_SI = 1.98847e30  # kg
Mpc_SI = 3.0856775815e22

# def point_mass_deflection(theta0, xi, chi_ls, chi_s, M, conf):
#     """
#     Deflection angle for point mass for flat universe.
    
#     Parameters
#     ----------
#     theta0 : float
#         Initial position of the ray, shape = [ray_num, 2], in radian
#     xi : float
#         Impact parameter, distance of closest approach, shape = [ray_num, 2], in Mpc
#     chi_ls : float
#         Comoving distance from lens to source, shape = () or [ray_num], in Mpc
#     chi_s : float
#         Comoving distance to source, shape = () or [ray_num], in Mpc
#     M : float
#         Mass of the lens, shape = () or [ray_num], in M
        
#     Returns
#     -------
#     theta_s : float
#         Final position of the ray, shape = [ray_num, 2], in radians.
#     """
#     alpha = 4 * conf.G * M*0.3 / conf.c ** 2 / xi
#     print('max deflection in arcmin', jnp.max(chi_ls / chi_s * alpha) * 3437.75)
#     theta_s = theta0 - chi_ls / chi_s * alpha
#     return theta_s 
    
# def impact_parameter(theta0, chi_l):
#     """
#     Impact parameter for point mass for flat universe.
#     The lens is placed the center in the image plane.
#     """
#     xi = theta0 * chi_l
#     return xi

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
        
    return theta_s 

def visual_ray_point_mass(theta0, theta_prev, theta_s, chi_l, chi_s, M, conf, ptcl):
    """
    Visualize the ray positions on the image plane.
    Assume source at chi_s and theta_s is the inferred true angular position.
    """
    
    
    # x = jnp.arange(conf.mesh_shape[0])*conf.cell_size
    # x -= x.mean()
    # y = jnp.arange(conf.mesh_shape[1])*conf.cell_size
    # y -= y.mean()
    # z = jnp.arange(conf.mesh_shape[2])*conf.cell_size

    # coord3D = ptcl.pos()[0]-jnp.array(conf.box_size)/2
    # coord2D = coord3D[0:2]/coord3D[2]
    
    
    theta_theory = point_mass_deflection(theta0, chi_l, chi_s, M, conf)
    
    # convert to arcmin
    theta0 = theta0 * 3437.75
    theta_prev = theta_prev * 3437.75
    theta_s = theta_s * 3437.75
    theta_E = jnp.sqrt((chi_s-chi_l) / (chi_s) / chi_l * 4 * conf.G * M / conf.c**2) * 3437.75

    if chi_s > 800:
        f,axs = plt.subplots(1,2,figsize=(9,4.5))
        # plot theta_prev, theta_s, theta_theory
        # axs[0].scatter(coord2D[0],coord2D[1],s=16,color='k')
        # plot x = y = 0 lines
        # draw einstein radius
        axs[0].add_patch(plt.Circle((0,0), theta_E, fill=False, color='k'))
        axs[0].axvline(0, color='k', linestyle='--')
        axs[0].axhline(0, color='k', linestyle='--')
        
        # create mask such that absolute magnitude of theta < theta_E
        mask = jnp.linalg.norm(theta_theory,axis=-1) > theta_E
        # join with mask with theta_s
        mask = mask & (jnp.linalg.norm(theta_s,axis=-1) > theta_E)
        # print masked out numbers
        print('masked out', jnp.sum(~mask))
        theta_prev = theta_prev[mask]
        theta_s = theta_s[mask]
        theta_theory = theta_theory[mask]
        # axs[0].scatter(theta_prev[...,0],theta_prev[...,1],s=1,color='k',alpha=0.5)
        axs[0].scatter(theta_s[...,0],theta_s[...,1],s=1,color='r',label='ray-traced')
        axs[0].scatter(theta_theory[...,0],theta_theory[...,1],s=1,color='b',label='theory')
        # axs[0].scatter(theta0[...,0],theta0[...,1],s=1,color='b')
        axs[0].set_aspect('equal')

        axs[0].set_xlim(-250,250)
        axs[0].set_ylim(-250,250)
        # legend
        axs[0].legend()
        
        # axs[0].set_xlim(-80,80)
        # axs[0].set_ylim(-80,80)
        # axs[0].set_xlim(-25,25)
        # axs[0].set_ylim(-25,25)
        # scatter plot of radius vs. residual angle
        r = jnp.linalg.norm(theta_theory,axis=-1)
        residual = jnp.linalg.norm(theta_s - theta_theory,axis=-1)
        axs[1].scatter(r,residual,s=1,color='k')
        axs[1].set_xlabel('radius (arcmin)')
        axs[1].set_ylabel('absolute residual (arcmin) ')
        axs[1].set_title('between RT and point mass equation')
        plt.show()
    return
    
    # for i in range(theta0.shape[0]):
    #     # very small arows
    #     axs[1].arrow(theta0[i,0],theta0[i,1],xi[i,0],xi[i,1],fc='b',ec='b')
    # # aspect ratio
    # axs[0].set_aspect('equal')
    # axs[1].set_aspect('equal')
    # plt.show()
    # return

# def visual_ray(pos_ip1,pos_ip2,disp):
#     # pos_ip1 shape = [ray_num, 2]
#     f,axs = plt.subplots(1,2,figsize=(8,4))
#     axs[0].scatter(pos_ip1[...,0],pos_ip1[...,1],s=1,color='k')
#     axs[0].scatter(pos_ip2[...,0],pos_ip2[...,1],s=1,color='r')
#     # plot disp centered on pos_ip1 with arrows
#     for i in range(pos_ip1.shape[0]):
#         # very small arows
#         axs[1].arrow(pos_ip1[i,0],pos_ip1[i,1],disp[i,0],disp[i,1],fc='b',ec='b')
#     # aspect ratio
#     axs[0].set_aspect('equal')
#     axs[1].set_aspect('equal')
#     plt.show()
#     return 