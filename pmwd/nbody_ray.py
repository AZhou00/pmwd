from functools import partial

from jax import value_and_grad, jit, vjp, custom_vjp
import jax.numpy as jnp

from pmwd import scatter
from pmwd.ray_mesh import compute_ray_mesh
from pmwd.boltzmann import growth, chi_a, r_a
from pmwd.lensing import grad_phi, lensing
from pmwd.nbody import nbody_init, nbody_step
from pmwd.ray_util import visual_ray_point_mass, lens_plane, lens_plane_nonperiodic
import matplotlib.pyplot as plt


def force_ray(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf):
    """deta on rays."""
    deta, dB = lensing(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf)
    return ray.replace(deta=deta, dB=dB)


def kick_ray(ray):
    """Kick."""
    eta = ray.eta + ray.deta
    B = ray.B + ray.dB
    return ray.replace(eta=eta, B=B)


def drift_ray(a_vel, a_prev, a_next, ray, cosmo, conf, ptcl):
    """
    Drift.
    factor = (chi_{n+1}-\chi_n)/r(\chi_{n+1/2})^2/c
    """
    factor = chi_a(a_next, cosmo, conf) - chi_a(a_prev, cosmo, conf)
    factor /= r_a(a_vel, cosmo, conf) ** 2
    factor /= conf.c

    # # point mass lens --------------------------------
    # chi_l = 770 #653.606 # in [L], i.e., Mpc
    # chi_s = chi_a(a_next, cosmo, conf)
    # if chi_s > chi_l:
    #     print( "chi_l", chi_l, "chi_s", chi_s,)
    #     chi_s_prev = chi_a(a_prev, cosmo, conf)
    #     M = cosmo.ptcl_mass * conf.ptcl_num * conf.mass_rescale #(conf.mesh_size / conf.ptcl_num) *
    #     visual_ray_point_mass(
    #         theta0=ray.pos_0(),
    #         theta_prev=ray.pos_ip(),
    #         theta_s=ray.pos_ip() + factor * ray.eta,
    #         chi_l=chi_l,
    #         chi_s_prev=chi_s_prev,
    #         chi_s=chi_s,
    #         M=M,
    #         conf=conf,
    #         ptcl=ptcl
    #         )
    # print("max drift [arcmin]", jnp.max(jnp.abs(ray.eta * factor) * 3437.75))
    # # ------------------------------------------------

    disp = ray.disp + factor * ray.eta
    A = ray.A + factor * ray.B
    return ray.replace(disp=disp, A=A)


def integrate_ray(a_prev, a_next, ptcl, ray, cosmo, conf):
    """Symplectic integration for one step."""
    # print('------------------')
    D = K = 0
    a_disp = a_vel = a_prev
    
    # potential for the ray evaluated at half step
    # a dependency only affects SO
    grad_phi3D = grad_phi((a_prev+a_next)/2, ptcl, cosmo, conf)
    
    # KDK
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ray = drift_ray(a_vel, a_disp, a_disp_next, ray, cosmo, conf, ptcl)
            a_disp = a_disp_next

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            a_c = (a_vel + a_vel_next) / 2
            ray = force_ray(a_vel, a_vel_next, a_c, ptcl, ray, grad_phi3D, cosmo, conf)
            ray = kick_ray(ray)
            a_vel = a_vel_next
    return ray


def nbody_ray_init(a, ray, obsvbl_ray, cosmo, conf):
    return ray, obsvbl_ray


def observe(ray, obsvbl_ray, cosmo, conf):
    """
    n(z) can come from redshift. 
    can add observable dependent n(z) later
    
    C_1 & = \frac{A_{11} + A_{22}}{2} \,, \\
    C_2 & = \frac{A_{12} - A_{21}}{2} \,, \\
    C_3 & = \frac{A_{22} - A_{11}}{2} \,, \\
    C_4 & = \frac{A_{12} + A_{21}}{2} \,. \\
    \kappa   & = 1 - C_1\,,               \\
    \omega   & = \frac{C_2}{1-\kappa} \,, \\
    \gamma_1 & = C_3 + C_4~ \omega \,,    \\
    \gamma_2 & = - C_4 + C_3~ \omega \,,  \\
    """
    # A & B have shape (ray_num, 2, 2)
    A, B = ray.A, ray.B
    # observables have shape (ray_num,)
    # kappa, gamma1, gamma2, omega = obsvbl_ray

    C1 = (A[..., 0, 0] + A[..., 1, 1]) / 2
    C2 = (A[..., 0, 1] - A[..., 1, 0]) / 2
    C3 = (A[..., 1, 1] - A[..., 0, 0]) / 2
    C4 = (A[..., 0, 1] + A[..., 1, 0]) / 2
    kappa = 1 - C1
    omega = C2 / (1 - kappa)
    gamma1 = C3 + C4 * omega
    gamma2 = -C4 + C3 * omega

    # n(z)
    kappa *= 1
    omega *= 1
    gamma1 *= 1
    gamma2 *= 1

    obsvbl_ray = kappa, gamma1, gamma2, omega
    return obsvbl_ray


def nbody_ray_step(a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf):
    ray = integrate_ray(a_prev, a_next, ptcl, ray, cosmo, conf)
    obsvbl_ray = observe(ray, obsvbl_ray, cosmo, conf)
    return ray, obsvbl_ray


def nbody_ray(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf, static=False):
    """
    N-body time integration with ray tracing updates.
    ray tracing nbody integration goes backward by default
    """
    from pmwd.gravity import laplace
    from pmwd.pm_util import fftfreq, fftfwd, fftinv

    # a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody = conf.a_nbody_ray  # in reverse order up the maximum source redshift

    if not static:
        # initialize the acceleration to ptcl does not do anything else.
        ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)
    # does nothing right now
    ray, obsvbl_ray = nbody_ray_init(a_nbody[0], ray, obsvbl_ray, cosmo, conf)

    for a_prev, a_next in zip(a_nbody[:-1], a_nbody[1:]):
        ray, obsvbl_ray = nbody_ray_step(
            a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf
        )
        
        # # compute potential field
        # val = jnp.zeros(conf.ptcl_num, dtype=conf.float_dtype)
        # val = val.at[0].set(conf.point_source_mass)
        # val /= conf.ptcl_cell_vol # point_source_mass in unit of rho_crit * Omega_m * 1Mpc^3
        # val *= conf.mesh_size / conf.ptcl_num
        # dens = scatter(ptcl, conf, val=val)
        # dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)
        # kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype) # has unit    
        # dens = fftfwd(dens)  # normalization canceled by that of irfftn below
        # pot = laplace(kvec, dens, cosmo) # mesh shape in Fourier space
        # pot = fftinv(pot, shape=conf.mesh_shape)
        # # project potential field and plot
        # chi_i = chi_a(a_prev, cosmo, conf)
        # chi_f = chi_a(a_next, cosmo, conf)
        # ray_cell_size, ray_mesh_shape = compute_ray_mesh(chi_i, chi_f, conf)
        # lens_mesh2D = lens_plane_nonperiodic(None, pot, conf, chi_i, chi_f, ray_mesh_shape, ray_cell_size, cosmo)
        # print('chi_i', chi_i, 'chi_f', chi_f)
        # c = plt.imshow(lens_mesh2D)
        # plt.colorbar(c)
        # plt.show()
        
        # k2 = sum(k**2 for k in kvec)
        # k2_zeros = jnp.where(k2 <= 1e-3, 1, 0)
        # print(k2.shape)
        # print('min', jnp.min(k2), 'max', jnp.max(k2))
        # c = plt.imshow(k2_zeros.sum(axis=2))
        # plt.colorbar(c)
        # plt.xlim(-10,270)
        # plt.ylim(-10,270)
        # plt.show()
        if not static:
            ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
    
    return ptcl, ray, obsvbl, obsvbl_ray


def nbody_ray_vis(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf, reverse=True, folder=''):
    """
    N-body time integration with ray tracing updates.
    ray tracing nbody integration goes backward by default
    """
    import os
    import numpy as np
    from pmwd import scatter
    
    # a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody = conf.a_nbody_ray  # in reverse order up the maximum source redshift

    # initialize the acceleration to ptcl does not do anything else.
    ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)
    # does nothing right now
    ray, obsvbl_ray = nbody_ray_init(a_nbody[0], ray, obsvbl_ray, cosmo, conf)

    for a_prev, a_next in zip(a_nbody[:-1], a_nbody[1:]):
        ray, obsvbl_ray = nbody_ray_step(
            a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf
        )
        ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
        
        filename = os.path.join(folder, f'{(a_prev+a_next)/2:.5f}.npz')
        dens = scatter(ptcl, conf)
        chi_i = chi_a(a_prev, cosmo, conf)
        chi_f = chi_a(a_next, cosmo, conf)
        ray_cell_size, ray_mesh_shape = compute_ray_mesh(chi_i, chi_f, conf)
        lens_mesh3D, lens_mesh2D = lens_plane(20, dens, conf, chi_i, chi_f, ray_mesh_shape, ray_cell_size, cosmo)
            
        plt.style.use('dark_background')
        f,ax = plt.subplots(1,1,figsize=(6,6))
        x,y = np.arange(ray_mesh_shape[0]), np.arange(ray_mesh_shape[1])
        x,y = x*ray_cell_size*180/np.pi*60, y*ray_cell_size*180/np.pi*60
        x,y = x-ray_mesh_shape[0]*ray_cell_size/2*180/np.pi*60, y-ray_mesh_shape[1]*ray_cell_size/2*180/np.pi*60
        xx,yy = np.meshgrid(x,y,indexing='ij')
        ax.pcolormesh(xx,yy,lens_mesh2D)
        
        factor = chi_a(a_next, cosmo, conf) - chi_a(a_prev, cosmo, conf)
        factor /= r_a((a_next+a_prev)/2, cosmo, conf) ** 2
        factor /= conf.c
        disp_vector = factor * ray.eta
        disp_vector = disp_vector.reshape(conf.ray_grid_shape[0], conf.ray_grid_shape[1], 2)
        x,y = np.arange(conf.ray_grid_shape[0]), np.arange(conf.ray_grid_shape[1])
        x,y = x*conf.ray_spacing*180/np.pi*60, y*conf.ray_spacing*180/np.pi*60
        x,y = x-conf.ray_grid_shape[0]*conf.ray_spacing/2*180/np.pi*60, y-conf.ray_grid_shape[1]*conf.ray_spacing/2*180/np.pi*60
        xx,yy = np.meshgrid(x,y,indexing='ij')
        # apply thinning to the vector field
        thinning = 20
        xx,yy = xx[::thinning,::thinning],yy[::thinning,::thinning]
        disp_vector = disp_vector[::thinning,::thinning]
        
        # normalize arrow length
        length = 2.3636355559576856e-06 * 1e9
        print(np.max(length))
        length = np.max(length)
        ax.quiver(xx,yy,disp_vector[:,:,0],disp_vector[:,:,1],color='white',scale=1/length)

        
        fov = conf.ray_grid_shape[0]*conf.ray_spacing*180/np.pi*60
        ax.set_xlim(-fov/2,fov/2)
        ax.set_ylim(-fov/2,fov/2)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.xaxis.set_tick_params(color='white')
        ax.yaxis.set_tick_params(color='white')
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=1)
        ax.set_xlabel('arcmin',color='white',fontsize=14)
        ax.set_ylabel('arcmin',color='white',fontsize=14)
        # set tick labels to larfer
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        # set back ground color to black
        ax.set_facecolor('black')
        # set tick labels to white
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        # tick inwards for both x and y, larger
        ax.tick_params(axis='x',direction='in',length=6)
        ax.tick_params(axis='y',direction='in',length=6)
        label = np.where(conf.a_nbody_ray == a_prev)[0][0]
        plt.savefig(os.path.join('/ocean/projects/phy230060p/junzhez/pmwdray_test/blender_data/ray_tracing_lens_plane', f'{label:04d}.png'))
        plt.show()
        
        np.savez(
            filename,
            a=(a_prev+a_next)/2, 
            mesh_cell_size=conf.cell_size,
            mesh_shape=conf.mesh_shape,
            ptcl_spacing=conf.ptcl_spacing,
            ptcl_shape=conf.ptcl_grid_shape,
            ptcl_pos=ptcl.pos(),
            dens = dens,
            ray_pos2d=ray.pos_ip(),
            ray_pos3d=ray.pos_3d((a_prev+a_next)/2, cosmo, conf),
            obsvbl_ray = obsvbl_ray,
            lens_mesh3D = lens_mesh3D,
            lens_mesh2D = lens_mesh2D,
        )
    return ptcl, ray, obsvbl, obsvbl_ray


# def nbody_ray_point_mass(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf, reverse=True):
#     """
#     point mass lensing test
#     all point mass is placed at the center of the box ~ 400Mpc/h, and source plane at the end of the box
#         chi_l = conf.box_size[-1]/2 #Mpc/h
#         chi_s = conf.box_size[-1] #Mpc/h, approx. chi_a(conf.a_nbody_ray[-1], cosmo, conf)
#     """
#     from pmwd.ray_util import deflection_pointsources
#     a_nbody = conf.a_nbody_ray  # in reverse order up the maximum source redshift
#     ray, obsvbl_ray = nbody_ray_init(a_nbody[0], ray, obsvbl_ray, cosmo, conf)

#     for a_prev, a_next in zip(a_nbody[:-1], a_nbody[1:]):
#         ray, obsvbl_ray = nbody_ray_step(
#             a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf
#         )
#     return ptcl, ray, obsvbl, obsvbl_ray, data

#         chi_l = conf.point_source_chi_l #Mpc/h 
#         chi_s = chi_a(a_next, cosmo, conf)
#         M = cosmo.conf.rho_crit * cosmo.Omega_m * (1)**3 * conf.point_source_mass # point_source_mass in unit of rho_crit * Omega_m * 1Mpc^3
#         # cosmo.conf.rho_crit * cosmo.Omega_m * 1e4 * 4 * 1.665**3 #1e4 * 8 * 8 * 4 #cosmo.ptcl_mass * conf.ptcl_num * conf.mass_rescale# / 1.414
#         # print('mass', M)
#         if chi_s > conf.box_size[-1]*19/20:
#             # initial position
#             theta_0 = ray.pos_0() * 3437.75 #arcmin
            
#             # theory lensing
#             Lx, Ly  = conf.box_size[0], conf.box_size[1]

#             ms = jnp.array([[0, 0]])
#             theta_theory = theta_0 - deflection_pointsources(theta_0, ms, M, chi_s, chi_l, conf)
#             print(theta_theory[0], 'theory')
            
#             ms = jnp.array(
#                 [
#                     [0, 0],
#                     # ring 1
#                     [Lx, 0],
#                     [-Lx, 0],
#                     [0, Ly],
#                     [0, -Ly],
#                     [Lx, Ly],
#                     [Lx, -Ly],
#                     [-Lx, Ly],
#                     [-Lx, -Ly],
#                     # ring 2
#                     [-2*Lx, 2*Ly],
#                     [-Lx, 2*Ly],
#                     [0, 2*Ly],
#                     [Lx, 2*Ly],
#                     [2*Lx, 2*Ly],
#                     [2*Lx, Ly],
#                     [2*Lx, 0],
#                     [2*Lx, -Ly],
#                     [2*Lx, -2*Ly],
#                     [Lx, -2*Ly],
#                     [0, -2*Ly],
#                     [-Lx, -2*Ly],
#                     [-2*Lx, -2*Ly],
#                     [-2*Lx, -Ly],
#                     [-2*Lx, 0],
#                     [-2*Lx, Ly],
    
#                 ]
#             )
#             theta_theory = theta_0 - deflection_pointsources(theta_0, ms, M, chi_s, chi_l, conf)
#             print(theta_theory[0], 'theory_periodic')

#             # HRT lensing
#             theta_s = ray.pos_ip() * 3437.75 #arcmin
            
#             # apply Einstein radius mask
#             scale = 4
#             theta_E = jnp.sqrt((chi_s-chi_l) / (chi_s) / chi_l * 4 * conf.G * M / conf.c**2) * 3437.75
#             mask = jnp.linalg.norm(theta_theory, axis=-1) > theta_E * scale
#             mask = mask & (jnp.linalg.norm(theta_s, axis=-1) > theta_E * scale)
#             mask = mask & (jnp.linalg.norm(theta_0, axis=-1) > theta_E * scale)
#             theta_0 = theta_0[mask]
#             theta_theory = theta_theory[mask]
#             theta_s = theta_s[mask]
#             data = (theta_0, theta_theory, theta_s, theta_E)
#             print(theta_s[0], 'HRT')

#             # print(chi_s)
#             # f,axs = plt.subplots(1,2,figsize=(12,6))
#             # def plot_ip_alpha_theory(ax, theta_0, theta_theory, theta_s, theta_E, limit=100, **kwargs):
#             #     error_x = (theta_s[:,0] - theta_theory[:,0])
#             #     error_y = (theta_s[:,1] - theta_theory[:,1])

#             #     alpha_theory = theta_theory - theta_0
#             #     alpha_theory = jnp.linalg.norm(alpha_theory, axis=-1)

#             #     alpha_s = theta_s - theta_0
#             #     alpha_s = jnp.linalg.norm(alpha_s, axis=-1)

#             #     error_rel = (alpha_s - alpha_theory) / alpha_theory

#             #     error_plot = (theta_theory - theta_0)[:,0]

#             #     ax.add_patch(plt.Circle((0,0), theta_E, fill=False, color='k', linewidth=1))
#             #     c = ax.scatter(theta_0[:,0], theta_0[:,1], c=error_plot, s=2, cmap='coolwarm', **kwargs)#,vmin=-0.05,vmax=0.05
#             #     plt.colorbar(c, ax=ax)
#             #     ax.set_xlim(-limit,limit)
#             #     ax.set_ylim(-limit,limit)
#             #     ax.set_xlabel(r'$\theta_x$ [arcmin]')
#             #     ax.set_ylabel(r'$\theta_y$ [arcmin]')
#             #     ax.set_aspect('equal')
#             #     ax.set_title(r'$\alpha_{theory,x}$')
#             #     return
#             # def plot_ip_alpha_s(ax, theta_0, theta_theory, theta_s, theta_E, limit=100, **kwargs):
#             #     error_x = (theta_s[:,0] - theta_theory[:,0])
#             #     error_y = (theta_s[:,1] - theta_theory[:,1])

#             #     alpha_theory = theta_theory - theta_0
#             #     alpha_theory = jnp.linalg.norm(alpha_theory, axis=-1)

#             #     alpha_s = theta_s - theta_0
#             #     alpha_s = jnp.linalg.norm(alpha_s, axis=-1)

#             #     error_plot = (theta_s - theta_0)[:,0]

#             #     ax.add_patch(plt.Circle((0,0), theta_E, fill=False, color='k', linewidth=1))
#             #     c = ax.scatter(theta_0[:,0], theta_0[:,1], c=error_plot, s=2, cmap='coolwarm', **kwargs)#,vmin=-0.05,vmax=0.05
#             #     plt.colorbar(c, ax=ax)
#             #     ax.set_xlim(-limit,limit)
#             #     ax.set_ylim(-limit,limit)
#             #     ax.set_xlabel(r'$\theta_x$ [arcmin]')
#             #     ax.set_ylabel(r'$\theta_y$ [arcmin]')
#             #     ax.set_aspect('equal')
#             #     ax.set_title(r'$\alpha_{HRT,x}$')
#             #     return
#             # plot_ip_alpha_theory(axs[0], *data, limit=100,vmin=-8,vmax=8)
#             # plot_ip_alpha_s(axs[1], *data, limit=100,vmin=-8,vmax=8)
#             plt.show()