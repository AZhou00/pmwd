import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
from pmwd import scatter
from pmwd.boltzmann import chi_a
from pmwd.lensing import mesh3D_coord, project
import numpy as np
from jax import jit

# https://github.com/lenstronomy/lenstronomy-tutorials/blob/main/Notebooks/GettingStarted/starting_guide.ipynb
# https://colab.research.google.com/github/sibirrer/strong_lensing_lectures/blob/main/Lectures/lensing_basics_I.ipynb

def deflection_pointsources(theta, ms, M, chi_s, chi_l, conf):
    # Constants: Convert arcmin to radians for calculations
    arcmin_to_rad = jnp.pi / 180 / 60
    G,c = conf.G,conf.c
    
    # Convert theta from arcmin to radians
    theta = jnp.array(theta) * arcmin_to_rad
    ms = jnp.array(ms)

    
    # Initialize deflection to zero
    alpha = jnp.zeros_like(theta)
    
    # Compute gravitational deflection for each point mass
    for m in ms:
        b = theta * chi_l
        v = b-m
        v_norm = jnp.linalg.norm(v, axis=-1)[..., None]
        
        alpha_hat = 4 * G * M / (c**2 * v_norm) * (v / v_norm)
        scaled_alpha = (chi_s - chi_l) / (chi_s) * alpha_hat
        alpha += scaled_alpha

    # Convert deflection from radians back to arcminutes
    alpha_in_arcmin = alpha / arcmin_to_rad
    return alpha_in_arcmin


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

def lens_plane_diagram(cosmo, conf, chi_l=None, name=None):
    from matplotlib.patches import Arc

    f, ax = plt.subplots(1,1,figsize=(16,5))

    xmin = 0
    xmax = conf.box_size[2]
    ymin = -conf.box_size[0]/2
    ymax = conf.box_size[0]/2

    chi_s = chi_a(conf.a_nbody_rt[-1], cosmo, conf)

    # box
    xmin = 0
    xmax = conf.box_size[2]
    n = 0
    while xmax <= chi_s+conf.box_size[2]:
        ax.plot([xmin,xmin,xmax,xmax,xmin], [ymin,ymax,ymax,ymin,ymin], '#333333', lw=2, alpha=1)
        xmin += conf.box_size[2]
        xmax += conf.box_size[2]
        n += 1

    ip_fov = conf.ray_spacing * conf.ray_grid_shape[0] # in radians
    theta_min = -ip_fov / 2
    theta_max = ip_fov / 2
    # light cone

    # lens planes
    lens_color = ['#8ecae6', '#fcbf49']
    a_lens_planes = np.concatenate([conf.a_nbody_rt, (conf.a_nbody_rt[1:] + conf.a_nbody_rt[:-1]) / 2,])
    a_lens_planes = np.sort(a_lens_planes)
    chi_lens = chi_a(a_lens_planes, cosmo, conf)
    n_lens = 0
    for chi in chi_lens[:-1]:
        n_lens += 1

        if n_lens % 2 == 0:
            color = lens_color[0]
            ax.add_line(
                plt.Line2D(
                    [chi, chi],
                    [theta_max * chi, conf.box_size[0]*2/3],
                    color='k',
                    linestyle=":",
                    lw=1,
                )
            )
            ax.text(
                chi,
                conf.box_size[0] * 2 / 3 * 1.07,
                f"{chi:.0f}",
                ha="center",
                va="bottom",
                rotation=65,
                fontsize=11,
            )  # $[\mathrm{{Mpc}}]$

        else:
            color = lens_color[1]
            ax.add_line(
                plt.Line2D(
                    [chi, chi],
                    [-conf.box_size[0]*2/3, theta_min * chi, ],
                    color='k',
                    linestyle=":",
                    lw=1,
                )
            )
            ax.text(
                chi,
                - conf.box_size[0] * 2 / 3 * 1.6,
                f"{chi:.0f}",
                ha="center",
                va="bottom",
                rotation=65,
                fontsize=11,
            )  # $[\mathrm{{Mpc}}]$
            
        ax.add_line(
            plt.Line2D(
                [chi, chi],
                [theta_min * chi, theta_max * chi],
                color=color,
                linestyle="-",
                lw=1.3,
            )
        )

    # arc
    light_cone_color = '#c1121f'
    ax.add_patch(
        Arc(
            (0, 0),
            2 * chi_s,
            2 * chi_s,
            angle=0,
            theta1=theta_min * 180 / np.pi,
            theta2=theta_max * 180 / np.pi,
            color=light_cone_color,
            lw=2,
        )
    )
    # lines
    x1 = chi_s * np.cos(theta_min)
    x2 = chi_s * np.cos(theta_max)
    y1 = chi_s * np.sin(theta_min)
    y2 = chi_s * np.sin(theta_max)
    ax.add_line(plt.Line2D([0, x1], [0, y1], color=light_cone_color,lw=2,))
    ax.add_line(plt.Line2D([0, x1], [0, y2], color=light_cone_color,lw=2,))

    if chi_l is not None:
        # plot lensing mass
        ax.plot(chi_l, 0, 'kx')

    ax.set_aspect('equal')
    ax.set_xlim(-75, n*conf.box_size[2]+75)
    ax.set_ylim(-conf.box_size[0]*1.25, conf.box_size[0]*1.25)
    ax.set_xlabel(r'$\chi$ [Mpc]', fontsize=14)
    ax.set_ylabel(r'$\chi$ [Mpc]', fontsize=14)
    # tick size
    ax.tick_params(axis='both', which='major', labelsize=12)
    # grid lines
    ax.grid(which='both')
    if name is not None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def lens_plane_nonperiodic(lens_thickness, dens, conf, chi_i, chi_f, ray_mesh_shape, ray_cell_size, cosmo):
    coord3D, z, r = mesh3D_coord(chi_i, chi_f, cosmo, conf)
    kernel = jnp.where((z >= chi_i) & (z <= chi_f), 1.0, 0.0)
    kernel *= jnp.where(z > 0, 1.0, 0.0)
    dens_vis = jnp.einsum('xyz,z->xyz', dens, kernel)
    dens_vis = dens_vis.reshape(-1, 1)
    lens_mesh2D = project(
            val_mesh3D=dens_vis,
            coord3D=coord3D,
            mesh2D_mesh_shape=ray_mesh_shape,
            mesh2D_cell_size=ray_cell_size,
            conf=conf,
        )
    return lens_mesh2D[...,0]
    # return dens_vis.sum(axis=2)

def lens_plane(lens_thickness, dens, conf, chi_i, chi_f, ray_mesh_shape, ray_cell_size, cosmo):
    # dens is a Nx, Ny, Nz array of density field, with periodic boundary condition
    # z has shape Nz labels the comoving distance chi value in z direction
    coord3D, z, r = mesh3D_coord(chi_i, chi_f, cosmo, conf)
    
    chi_mid = (chi_i + chi_f) / 2
    # index of lens plabe closest to chi_mid
    j = jnp.argmin(jnp.abs(z - chi_mid))
    # select +- lens_thickness many planes from the z that is closest to chi_mid, with periodic boundary condition
    # start from the smaller end of z
    # total length in z direction of the lens_mesh3D = 2*lens_thickness + 1
    if j < lens_thickness:
        # then the tail is long enough to reach the end of z
        lens_mesh3D = jnp.concatenate([dens[..., j - lens_thickness:], dens[..., :j + lens_thickness + 1]], axis=-1)
        # print('j < lens_thickness ,slices have length',dens[..., j - lens_thickness:].shape,dens[..., :j + lens_thickness + 1].shape)
    else:
        if j + lens_thickness + 1 > len(z):
            # then the head is long enough to reach the beginning of z
            # print(len(z) - j + lens_thickness)
            lens_mesh3D = jnp.concatenate([dens[..., j - lens_thickness:], dens[..., :lens_thickness - (len(z) - j)+1]], axis=-1)
            # print('j + lens_thickness + 1 > len(z),slices have length',dens[..., j - lens_thickness:].shape,dens[..., :lens_thickness - (len(z) - j)+1].shape)
        else:
            lens_mesh3D = dens[..., j - lens_thickness : j + lens_thickness + 1]
            # print('else, slices have length',dens[..., j - lens_thickness : j + lens_thickness + 1].shape)
    assert lens_mesh3D.shape[-1] == 2*lens_thickness + 1, f"lens_mesh3D.shape[-1] = {lens_mesh3D.shape[-1]} != 2*lens_thickness + 1"
    
    # project the density plane
    kernel = jnp.where((z >= chi_i) & (z <= chi_f), 1, 0)
    kernel *= jnp.where(z > 0, 1, 0)
    dens_vis = jnp.einsum('xyz,z->xyz', dens, kernel)
    dens_vis = dens_vis.reshape(-1, 1)
    lens_mesh2D = project(
            val_mesh3D=dens_vis,
            coord3D=coord3D,
            mesh2D_mesh_shape=ray_mesh_shape,
            mesh2D_cell_size=ray_cell_size,
            conf=conf,
        )
    return lens_mesh3D, lens_mesh2D[...,0]


#     lens_mesh2D = project(
#         val_mesh3D=lens_mesh3D,
#         coord3D=coord3D,
#         mesh2D_mesh_shape=ray_mesh_shape,
#         mesh2D_cell_size=ray_cell_size,
#         conf=conf,
#     )


#     project(
#     val_mesh3D,
#     coord3D,
#     mesh2D_cell_size,
#     mesh2D_mesh_shape,
#     conf,
# )

#     coord3D

#     lens_mesh3D =
#     lens_mesh2D =

#     f,axs = plt.subplots(2,3,figsize=(36,24),sharex=True,sharey=True)


#     # # plot the comoving space projection of the density field
#     # kernel = jnp.where((chi >= chi_i) & (chi <= chi_f), 1, 0)
#     # kernel *= jnp.where(chi > 0, 1, 0)
#     # print('number of planes in projection:',kernel.sum())
#     # dens_vis = scatter(ptcl, conf)
#     # dens_vis = jnp.einsum("xyz,z->xyz", dens_vis, kernel)
#     # dens_vis = dens_vis.sum(axis=2)
#     # xx, yy = jnp.meshgrid(x, y, indexing='ij')
#     # # log norm with cut off
#     # print('max dens:',dens_vis.max(), 'min dens:',dens_vis.min())
#     # axs[0].pcolormesh(xx, yy, dens_vis, norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#     # axs[0].set_aspect('equal', adjustable='box')
#     # axs[0].set_xlabel('x [Mpc]')
#     # axs[0].set_ylabel('y [Mpc]')
#     # axs[0].set_title('comoving density field projected along z',fontsize=20)

#     # set up image plane coords
#     # project density field to the image plane
#     x = jnp.arange(ray_mesh_shape[0]) * ray_cell_size
#     x -= x.mean()
#     y = jnp.arange(ray_mesh_shape[1]) * ray_cell_size
#     y -= y.mean()
#     xx, yy = jnp.meshgrid(x, y, indexing="ij")
#     xx *= 180 / jnp.pi * 60
#     yy *= 180 / jnp.pi * 60
#     # field of view
#     fov = [
#         conf.ray_grid_shape[0] * conf.ray_spacing * 180 / jnp.pi * 60,
#         conf.ray_grid_shape[1] * conf.ray_spacing * 180 / jnp.pi * 60,
#     ]
#     fov_padded = [
#         ray_mesh_shape[0] * ray_cell_size * 180 / jnp.pi * 60,
#         ray_mesh_shape[1] * ray_cell_size * 180 / jnp.pi * 60,
#     ]
#     print("field of view [arcmin]:", fov[0], fov[1])
#     print("padded field of view [arcmin]:", fov_padded[0], fov_padded[1])
#     # quiver settings
#     skip = (slice(None, None, 2), slice(None, None, 2))

#     # plot light cone density
#     dens_vis = scatter(ptcl, conf)
#     kernel = jnp.where((chi >= chi_i) & (chi <= chi_f), 1, 0)
#     kernel *= jnp.where(chi > 0, 1, 0)
#     dens_vis = jnp.einsum('xyz,z->xyz', dens_vis, kernel)
#     dens_vis = dens_vis.reshape(-1, 1)
#     dens_vis = project(
#             val_mesh3D=dens_vis,
#             coord3D=coord3D,
#             mesh2D_mesh_shape=ray_mesh_shape,
#             mesh2D_cell_size=ray_cell_size,
#             conf=conf,
#         )
#     axs[0,0].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#     axs[1,0].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#     # # quiver plot of defl_2D
#     axs[1,0].quiver(
#         xx[skip],
#         yy[skip],
#         defl_2D_smth[..., 0][skip],
#         defl_2D_smth[..., 1][skip],
#         color="w",
#         scale=2e4,
#     )

#     # plot nonsmoothed deta field
#     axs[0,1].pcolormesh(xx, yy, defl_2D[...,0])
#     axs[1,1].pcolormesh(xx, yy, defl_2D[...,1])

#     axs[0,2].pcolormesh(xx, yy, defl_2D_smth[...,0])
#     axs[1,2].pcolormesh(xx, yy, defl_2D_smth[...,1])

#     axs[0,1].set_title('deta x',fontsize=20)
#     axs[1,1].set_title('deta y',fontsize=20)
#     axs[0,2].set_title('smooth & deconv deta x',fontsize=20)
#     axs[1,2].set_title('smooth & deconv deta y',fontsize=20)


#     for ax in axs.flatten():
#         ax.set_aspect('equal', adjustable='box')
#         ax.plot([-fov[0] / 2, fov[0] / 2], [-fov[1] / 2, -fov[1] / 2], 'w-')
#         ax.plot([-fov[0] / 2, fov[0] / 2], [fov[1] / 2, fov[1] / 2], 'w-')
#         ax.plot([-fov[0] / 2, -fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#         ax.plot([fov[0] / 2, fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#         # ax.set_xlim(-fov[0] / 2, fov[0] / 2)
#         # ax.set_ylim(-fov[1] / 2, fov[1] / 2)
#         ax.set_xlabel('x [arcmin]')
#         ax.set_ylabel('y [arcmin]')

#     plt.show()

#     if diagnostic:
#         # x_ptcl, y_ptcl, z_ptcl = ptcl.pos().T
#         # x_ray, y_ray, z_ray = ray.pos_3d(a_i, cosmo, conf, wrap=True).T
#         # f = plt.figure(figsize=(15, 15))
#         # f, ax = snapshot(f, ptcl, ray, a_i, cosmo, conf,  wrap=True,elev=-210, azim=-25, roll=80)
#         # plt.show()
#         # plt.close(f)

#         f,axs = plt.subplots(1,3,figsize=(36,12))

#         # plot the comoving space projection of the density field
#         kernel = jnp.where((z >= chi_i) & (z <= chi_f), 1, 0)
#         kernel *= jnp.where(z > 0, 1, 0)
#         print('number of planes in projection:',kernel.sum())
#         dens_vis = scatter(ptcl, conf)
#         dens_vis = jnp.einsum("xyz,z->xyz", dens_vis, kernel)
#         dens_vis = dens_vis.sum(axis=2)
#         xx, yy = jnp.meshgrid(x, y, indexing='ij')
#         # log norm with cut off
#         print('max dens:',dens_vis.max(), 'min dens:',dens_vis.min())
#         axs[0].pcolormesh(xx, yy, dens_vis, norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#         axs[0].set_aspect('equal', adjustable='box')
#         axs[0].set_xlabel('x [Mpc]')
#         axs[0].set_ylabel('y [Mpc]')
#         axs[0].set_title('comoving density field projected along z',fontsize=20)

#         # project density field to the image plane
#         x = jnp.arange(ray_mesh_shape[0]) * ray_cell_size
#         x -= x.mean()
#         y = jnp.arange(ray_mesh_shape[1]) * ray_cell_size
#         y -= y.mean()
#         xx, yy = jnp.meshgrid(x, y, indexing="ij")
#         xx *= 180 / jnp.pi * 60
#         yy *= 180 / jnp.pi * 60
#         # field of view
#         fov = [
#             conf.ray_grid_shape[0] * conf.ray_spacing * 180 / jnp.pi * 60,
#             conf.ray_grid_shape[1] * conf.ray_spacing * 180 / jnp.pi * 60,
#         ]
#         fov_padded = [
#             ray_mesh_shape[0] * ray_cell_size * 180 / jnp.pi * 60,
#             ray_mesh_shape[1] * ray_cell_size * 180 / jnp.pi * 60,
#         ]
#         print("field of view [arcmin]:", fov[0], fov[1])
#         print("padded field of view [arcmin]:", fov_padded[0], fov_padded[1])
#         skip = (slice(None, None, 2), slice(None, None, 2))

#         # plot
#         dens_vis = scatter(ptcl, conf)
#         kernel = jnp.where((z >= chi_i) & (z <= chi_f), 1, 0)
#         kernel *= jnp.where(z > 0, 1, 0)
#         dens_vis = jnp.einsum('xyz,z->xyz', dens_vis, kernel)
#         dens_vis = dens_vis.reshape(-1, 1)
#         dens_vis = project(
#                 val_mesh3D=dens_vis,
#                 coord3D=coord3D,
#                 mesh2D_mesh_shape=ray_mesh_shape,
#                 mesh2D_cell_size=ray_cell_size,
#                 conf=conf,
#             )

#         axs[1].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#         # quiver plot of defl_2D
#         skip = (slice(None, None, 2), slice(None, None, 2))
#         axs[1].quiver(
#             xx[skip],
#             yy[skip],
#             defl_2D[..., 0][skip],
#             defl_2D[..., 1][skip],
#             color="w",
#             scale=2e4,
#         )
#         axs[1].set_aspect("equal", adjustable="box")
#         axs[1].plot([-fov[0] / 2, fov[0] / 2], [-fov[1] / 2, -fov[1] / 2], 'w-')
#         axs[1].plot([-fov[0] / 2, fov[0] / 2], [fov[1] / 2, fov[1] / 2], 'w-')
#         axs[1].plot([-fov[0] / 2, -fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#         axs[1].plot([fov[0] / 2, fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#         # axs[1].set_xlim(-fov[0] / 2, fov[0] / 2)
#         # axs[1].set_ylim(-fov[1] / 2, fov[1] / 2)
#         axs[1].set_xlabel('x [arcmin]')
#         axs[1].set_ylabel('y [arcmin]')
#         axs[1].set_title('deflection field & density angular proj',fontsize=20)

#         # plot the smoothed deflection field
#         axs[2].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#         # quiver plot of defl_2D
#         axs[2].quiver(
#             xx[skip],
#             yy[skip],
#             defl_2D_smth[..., 0][skip],
#             defl_2D_smth[..., 1][skip],
#             color="w",
#             scale=2e4,
#         )
#         axs[2].set_aspect("equal", adjustable="box")
#         axs[2].plot([-fov[0] / 2, fov[0] / 2], [-fov[1] / 2, -fov[1] / 2], 'w-')
#         axs[2].plot([-fov[0] / 2, fov[0] / 2], [fov[1] / 2, fov[1] / 2], 'w-')
#         axs[2].plot([-fov[0] / 2, -fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#         axs[2].plot([fov[0] / 2, fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#         # axs[1].set_xlim(-fov[0] / 2, fov[0] / 2)
#         # axs[1].set_ylim(-fov[1] / 2, fov[1] / 2)
#         axs[2].set_xlabel('x [arcmin]')
#         axs[2].set_ylabel('y [arcmin]')
#         axs[2].set_title('smoothed+deconv deflection field & density angular proj',fontsize=20)
#         plt.show()

#         f,axs = plt.subplots(1,2,figsize=(24,12))

#         axs[0].imshow(defl_2D[...,0])
#         axs[1].imshow(defl_2D_x)
#         plt.show()
