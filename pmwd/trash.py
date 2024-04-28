def gather_eta(ray_pos_ip, ray_pmid, defl_2D_smth, ray_cell_size, conf):
    """Compute the kick operator on the ray poasitions.

    Args:
        ray_pos_ip (_type_): shape (ray_num * 2,), flattened position array
        defl_2D_smth (_type_): shape (ray_mesh_shape[0], ray_mesh_shape[1], 2), smoothed deflection field
        ray_pmid (_type_): shape (ray_num, 2), ray.pmid
        ray_cell_size (_type_): _description_
        conf (_type_): _description_

    Returns:
        _type_: _description_
    """
    ray_pos_ip = ray_pos_ip.reshape(-1, 2)
    ray_mesh_shape = defl_2D_smth.shape[0:2]

    deta = _gather_rt(
        pmid = jnp.zeros_like(ray_pmid).astype(conf.pmid_dtype), 
        disp = ray_pos_ip, 
        conf = conf, 
        mesh = defl_2D_smth, 
        # val is 2D: deta in x and y
        val = jnp.zeros((conf.ray_num,2)), 
        offset = ray_mesh_center(ray_cell_size, ray_mesh_shape), 
        ray_cell_size = ray_cell_size,
        ray_mesh_shape = ray_mesh_shape
        )
    # visual_lensing(ptcl, defl_2D, defl_2D_smth, coord3D, conf, z, chi_i, chi_f, ray_mesh_shape, ray_cell_size)
    
    # print('deta', deta.shape) # (ray_num, 2)
    
    return deta.reshape(-1)


def lensing(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf):
    """    # Reshaping here follows this pattern below
        # A = jnp.arange(12).reshape(3,2,2)
        # print(A.shape)
        # print(A)
        # Ax = A[:,:,0]
        # Ay = A[:,:,1]
        # print()
        # print('Ax \n', Ax)
        # print('Ay \n', Ay)
        # Ax = Ax.reshape(-1)
        # Ay = Ay.reshape(-1)
        # print()
        # print('Ax \n', Ax)
        # print('Ay \n', Ay)
        # Ax = Ax.reshape(-1,2)
        # Ay = Ay.reshape(-1,2)
        # print()
        # print('Ax \n', Ax)
        # print('Ay \n', Ay)
        # A_new = jnp.stack((Ax,Ay), axis=-1)
        # print(jnp.allclose(A, A_new))

        # deta, f_jvp = linearize(f, ray.pos_ip() )
        # deta = deta.reshape(-1,2)
        # dB = f_jvp(ray.A)
        # print('dB', dB.shape) # (ray_num, 2)
    """
    
    r_i = distance_ad(a_i, cosmo, conf)
    r_f = distance_ad(a_f, cosmo, conf)
    ray_cell_size, ray_mesh_shape = compute_ray_mesh(r_i, r_f, conf)
    defl_2D_smth = deflection_field(a_i, a_f, a_c, ptcl, grad_phi3D, ray_cell_size, ray_mesh_shape, cosmo, conf)
    # print defl_2D_smth.shape # (ray_mesh_shape[0], ray_mesh_shape[1], 2)
    f = partial(
        gather_eta,
        ray_pmid=ray.pmid,
        defl_2D_smth=defl_2D_smth,
        ray_cell_size=ray_cell_size,
        conf=conf,
    )
    
    # first/second column of the distortion matrix (dtheta_n1_dtheta_0_x/y), 
    # flattened, shape = (ray_num * 2)
    
    # ray.pos_ip() shape = (ray_num, 2)
    # A.
    A0_x = ray.A[:,:,0].reshape(-1) 
    A0_y = ray.A[:,:,1].reshape(-1)
    # A0_x = ray.A[:,0,:].reshape(-1) 
    # A0_y = ray.A[:,1,:].reshape(-1)
    
    # ray positions, flattened, shape = (ray_num * 2)
    primals = ray.pos_ip().reshape(-1)
    
    # useful notes on linearize https://github.com/google/jax/issues/526
    # gradient, reshape results to (ray_num, 2)
    deta, f_jvp = linearize(f, primals) # deta' shape = 
    deta = deta.reshape(-1,2)
    dB_x = f_jvp(A0_x).reshape(-1,2)
    dB_y = f_jvp(A0_y).reshape(-1,2)
    # integers -> float0, others use jnp.zeros.
    # primals = (x,y,z)
    # f_jvp(x_v, jnp.zeros_like(y), jnp.zeros_like(z))
    
    # # stack two columns vector side by side
    dB = jnp.stack([dB_x, dB_y], axis=-1)
    
    return deta, dB

















# visualisation ------------------
# ------------------------------

def visual_nbody(ptcl,ray,a_i,cosmo,conf):
    x_ptcl, y_ptcl, z_ptcl = ptcl.pos().T
    x_ray, y_ray, z_ray = ray.pos_3d(a_i, cosmo, conf, wrap=True).T
    f = plt.figure(figsize=(15, 15))
    f, ax = snapshot(f, ptcl, ray, a_i, cosmo, conf,  wrap=True,elev=-210, azim=-25, roll=80)
    plt.show()
    plt.close(f)
    return


def visual_nbody_proj(ptcl, conf, x, y, chi, chi_i, chi_f):

    # plot the comoving space projection of the density field
    xx, yy = jnp.meshgrid(x, y, indexing="ij")

    kernel = jnp.where((chi >= chi_i) & (chi <= chi_f), 1, 0)
    kernel *= jnp.where(chi > 0, 1, 0)
    print("number of planes in projection:", kernel.sum())

    dens_vis = scatter(ptcl, conf)
    dens_vis = jnp.einsum("xyz,z->xyz", dens_vis, kernel)
    dens_vis = dens_vis.sum(axis=2)

    # log norm with cut off
    print("max dens:", dens_vis.max(), "min dens:", dens_vis.min())
    plt.show().pcolormesh(
        xx, yy, dens_vis, norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000)
    )
    plt.show().set_aspect("equal", adjustable="box")
    plt.show().set_xlabel("x [Mpc]")
    plt.show().set_ylabel("y [Mpc]")
    plt.show().set_title("comoving density field projected along z", fontsize=20)
    plt.show()

    return

def visual_density_defl2D(ptcl, defl_2D_smth, coord3D, conf, chi, chi_i, chi_f, ray_mesh_shape, ray_cell_size):
    f,axs = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
    
    # set up image plane coords
    x = jnp.arange(ray_mesh_shape[0]) * ray_cell_size
    x -= x.mean()
    y = jnp.arange(ray_mesh_shape[1]) * ray_cell_size
    y -= y.mean()
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    xx *= 180 / jnp.pi * 60
    yy *= 180 / jnp.pi * 60
    # field of view
    fov = [
        conf.ray_grid_shape[0] * conf.ray_spacing * 180 / jnp.pi * 60,
        conf.ray_grid_shape[1] * conf.ray_spacing * 180 / jnp.pi * 60,
    ]
    fov_padded = [
        ray_mesh_shape[0] * ray_cell_size * 180 / jnp.pi * 60,
        ray_mesh_shape[1] * ray_cell_size * 180 / jnp.pi * 60,
    ]
    print("field of view [arcmin]:", fov[0], fov[1])
    print("padded field of view [arcmin]:", fov_padded[0], fov_padded[1])
    # quiver settings
    skip = (slice(None, None, 2), slice(None, None, 2))

    # plot light cone density
    dens_vis = scatter(ptcl, conf)
    kernel = jnp.where((chi >= chi_i) & (chi <= chi_f), 1, 0)
    kernel *= jnp.where(chi > 0, 1, 0)
    dens_vis = jnp.einsum('xyz,z->xyz', dens_vis, kernel)
    dens_vis = dens_vis.reshape(-1, 1)
    dens_vis = project(
            val_mesh3D=dens_vis,
            coord3D=coord3D,
            mesh2D_mesh_shape=ray_mesh_shape,
            mesh2D_cell_size=ray_cell_size,
            conf=conf,
        )
    axs[0].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
    # # quiver plot of defl_2D
    axs[0].quiver(
        xx[skip],
        yy[skip],
        defl_2D_smth[..., 0][skip],
        defl_2D_smth[..., 1][skip],
        color="w",
        scale=2e4,
    )
    
    # plot nonsmoothed deta field
    axs[1].pcolormesh(xx, yy, defl_2D_smth[...,0])
    axs[2].pcolormesh(xx, yy, defl_2D_smth[...,1])
    
    axs[0].set_title('density field',fontsize=20)
    axs[1].set_title('deta x',fontsize=20)
    axs[2].set_title('deta y',fontsize=20)
    
    # set limits
    for ax in axs.flatten():
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-fov[0] / 2, fov[0] / 2)
        ax.set_ylim(-fov[1] / 2, fov[1] / 2)
        ax.set_xlabel('x [arcmin]')
        ax.set_ylabel('y [arcmin]')

    plt.show()


# def visual_lensing(ptcl, defl_2D, defl_2D_smth, coord3D, conf, chi, chi_i, chi_f, ray_mesh_shape, ray_cell_size):

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

# if diagnostic:
#     # x_ptcl, y_ptcl, z_ptcl = ptcl.pos().T
#     # x_ray, y_ray, z_ray = ray.pos_3d(a_i, cosmo, conf, wrap=True).T
#     # f = plt.figure(figsize=(15, 15))
#     # f, ax = snapshot(f, ptcl, ray, a_i, cosmo, conf,  wrap=True,elev=-210, azim=-25, roll=80)
#     # plt.show()
#     # plt.close(f)

#     f,axs = plt.subplots(1,3,figsize=(36,12))

#     # plot the comoving space projection of the density field
#     kernel = jnp.where((z >= chi_i) & (z <= chi_f), 1, 0)
#     kernel *= jnp.where(z > 0, 1, 0)
#     print('number of planes in projection:',kernel.sum())
#     dens_vis = scatter(ptcl, conf)
#     dens_vis = jnp.einsum("xyz,z->xyz", dens_vis, kernel)
#     dens_vis = dens_vis.sum(axis=2)
#     xx, yy = jnp.meshgrid(x, y, indexing='ij')
#     # log norm with cut off
#     print('max dens:',dens_vis.max(), 'min dens:',dens_vis.min())
#     axs[0].pcolormesh(xx, yy, dens_vis, norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#     axs[0].set_aspect('equal', adjustable='box')
#     axs[0].set_xlabel('x [Mpc]')
#     axs[0].set_ylabel('y [Mpc]')
#     axs[0].set_title('comoving density field projected along z',fontsize=20)

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
#     skip = (slice(None, None, 2), slice(None, None, 2))

#     # plot
#     dens_vis = scatter(ptcl, conf)
#     kernel = jnp.where((z >= chi_i) & (z <= chi_f), 1, 0)
#     kernel *= jnp.where(z > 0, 1, 0)
#     dens_vis = jnp.einsum('xyz,z->xyz', dens_vis, kernel)
#     dens_vis = dens_vis.reshape(-1, 1)
#     dens_vis = project(
#             val_mesh3D=dens_vis,
#             coord3D=coord3D,
#             mesh2D_mesh_shape=ray_mesh_shape,
#             mesh2D_cell_size=ray_cell_size,
#             conf=conf,
#         )

#     axs[1].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#     # quiver plot of defl_2D
#     skip = (slice(None, None, 2), slice(None, None, 2))
#     axs[1].quiver(
#         xx[skip],
#         yy[skip],
#         defl_2D[..., 0][skip],
#         defl_2D[..., 1][skip],
#         color="w",
#         scale=2e4,
#     )
#     axs[1].set_aspect("equal", adjustable="box")
#     axs[1].plot([-fov[0] / 2, fov[0] / 2], [-fov[1] / 2, -fov[1] / 2], 'w-')
#     axs[1].plot([-fov[0] / 2, fov[0] / 2], [fov[1] / 2, fov[1] / 2], 'w-')
#     axs[1].plot([-fov[0] / 2, -fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#     axs[1].plot([fov[0] / 2, fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#     # axs[1].set_xlim(-fov[0] / 2, fov[0] / 2)
#     # axs[1].set_ylim(-fov[1] / 2, fov[1] / 2)
#     axs[1].set_xlabel('x [arcmin]')
#     axs[1].set_ylabel('y [arcmin]')
#     axs[1].set_title('deflection field & density angular proj',fontsize=20)

#     # plot the smoothed deflection field
#     axs[2].pcolormesh(xx, yy, dens_vis.squeeze(), norm=matplotlib.colors.LogNorm(vmin=1, vmax=2000))
#     # quiver plot of defl_2D
#     axs[2].quiver(
#         xx[skip],
#         yy[skip],
#         defl_2D_smth[..., 0][skip],
#         defl_2D_smth[..., 1][skip],
#         color="w",
#         scale=2e4,
#     )
#     axs[2].set_aspect("equal", adjustable="box")
#     axs[2].plot([-fov[0] / 2, fov[0] / 2], [-fov[1] / 2, -fov[1] / 2], 'w-')
#     axs[2].plot([-fov[0] / 2, fov[0] / 2], [fov[1] / 2, fov[1] / 2], 'w-')
#     axs[2].plot([-fov[0] / 2, -fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#     axs[2].plot([fov[0] / 2, fov[0] / 2], [-fov[1] / 2, fov[1] / 2], 'w-')
#     # axs[1].set_xlim(-fov[0] / 2, fov[0] / 2)
#     # axs[1].set_ylim(-fov[1] / 2, fov[1] / 2)
#     axs[2].set_xlabel('x [arcmin]')
#     axs[2].set_ylabel('y [arcmin]')
#     axs[2].set_title('smoothed+deconv deflection field & density angular proj',fontsize=20)
#     plt.show()

#     f,axs = plt.subplots(1,2,figsize=(24,12))

#     axs[0].imshow(defl_2D[...,0])
#     axs[1].imshow(defl_2D_x)
#     plt.show()

# ------------------ lensing ------------------
# # query the lens plane
# slice_flag = False
# if slice_flag:
#     grad_mesh = slice(chi_i,grad_mesh,cosmo,conf)  # lens_mesh_shape + (3,)

#     # define lens plane mesh comoving coordinates,
#     # the particle mesh is defined in regular comoving distance grid
#     x = jnp.arange(conf.lens_mesh_shape[0])*conf.cell_size
#     y = jnp.arange(conf.lens_mesh_shape[1])*conf.cell_size
#     z = jnp.arange(conf.lens_mesh_shape[2])*conf.cell_size + chi_i #TODO chi_i?

#     # compute the 3D comoving coordinates of the potential mesh
#     # mimicking how the particle grid is set up
#     coords = jnp.meshgrid(*[x,y,z], indexing='ij')
#     coords = jnp.stack(coords, axis=-1).reshape(-1, conf.dim)  # shape (conf.lens_mesh_size, 3)
#     coord_z = coords[:,2] # shape (conf.lens_mesh_size,)
#     r = coord_z # TODO: add r(chi) function, maybe this can be cached
#     # print('shape of coord_z_angular_diam', coord_z_angular_diam.shape)

#     # compute the 2d image plane coordiante of each 3d mesh point
#     coords -= jnp.array([conf.ray_origin[0], conf.ray_origin[1], 0])
#     coords /= (r+1e-2)[:,jnp.newaxis] # shape (conf.lens_mesh_size, 3) TODO: stability at coord_z=0
#     # print(jnp.unique(coords))
#     coord2D = coords[:,0:2] # drop the z coords, shape (lens_mesh_size, 2)
#     # print(jnp.unique(coords))

# val_2d = project(val_mesh3D=grad_mesh, x=x, y=y, r=z, mesh2D_mesh_shape, mesh2D_cell_size, conf=conf, dim=conf.dim, )

# # compute lens kernel as a function radial comoving distance
# # TODO: add growth factor correction
# # for mesh point outside the lens plane, the kernel is set to 0
# lens_kernel = jnp.where((coord_z>=chi_i) & (coord_z<=chi_f), 2*r/conf.c, 0) # (conf.lens_mesh_size,)
# # scaling by cell
# lens_kernel *= jnp.where(r>0, conf.ptcl_cell_vol / conf.ray_spacing**2 / r**2, 0) # (conf.lens_mesh_size,)
# # lens_kernel /= jnp.where(r>0, conf.ptcl_cell_vol / conf.ray_spacing**2 / r**2, jnp.ones_like(coord_z)*1e6) # (conf.lens_mesh_size,)
# # print(lens_kernel)

# # apply the lens kernel
# grad_mesh = grad_mesh.reshape(-1, 3) # (lens_mesh_SHAPE + (3,)) -> (lens_mesh_size, 3)
# # only need grad in x & y directions
# grad_mesh = grad_mesh[:,0:2] # (lens_mesh_size, 2)
# grad_mesh = jnp.einsum('ij,i->ij', grad_mesh, lens_kernel) # (lens_mesh_size, 2)
# # print('shape of grad_mesh', grad_mesh.shape) # (lens_mesh_size, 2)
# # print('shape of coords', coords.shape) # (lens_mesh_size, 2)
# # print('mesh_shape', conf.ray_mesh_shape + grad_mesh.shape[1:]) # ray_mesh_shape + (2,)

# # diagnostics
# # plot density field in this slice
# dens_vis = jnp.where((coord_z>=chi_i) & (coord_z<=chi_f), jnp.ones_like(coord_z), 0)
# dens_vis = dens_vis.sum(axis=2)
# plt.imshow

# offset = -jnp.array([conf.ray_mesh_fov[0],conf.ray_mesh_fov[1]])/2 #TODO: is this right?
# # _scatter_rt is _scatter without conf.chunk_size dependence
# # ray_mesh_shape + (2,)
# grad_2d = _scatter_rt(
#     pmid=jnp.zeros_like(coords).astype(conf.pmid_dtype), # (lens_mesh_size, 2)
#     disp=coords, # (lens_mesh_size, 2)
#     conf=conf,
#     mesh=jnp.zeros(conf.ray_mesh_shape + grad_mesh.shape[1:], dtype=conf.float_dtype), # scatter to ray_mesh_shape
#     val = grad_mesh, # (lens_mesh_size, 2)
#     offset=offset,
#     cell_size=conf.ray_cell_size)
# print('shape of grad_2d after scatter', grad_2d.shape) # ray_mesh_shape + (2,)

# deconvolution and smoothing
# kvec_ray = fftfreq(conf.ray_mesh_shape, conf.ray_cell_size, dtype=conf.float_dtype)
# print('kvec_ray.shape', kvec_ray[0].shape,len(kvec_ray))

# # _gather_rt is _gather without conf.chunk_size dependence
# # (ray_num, 2)
# defl_2D = _gather_rt(
#     pmid = jnp.zeros_like(ray.pmid).astype(conf.pmid_dtype),
#     disp = ray.pos_ip(),
#     conf = conf,
#     mesh = defl_2D,
#     val = jnp.zeros((conf.ray_num,2)),
#     # offset = offset,
#     cell_size = conf.ray_cell_size)
# print('shape of grad_2d after gather', grad_2d.shape)

# return defl_2D

# def slice(chi_i,grad_mesh,cosmo,conf):
#     # take a slice (of fixed size) of the potential gradient mesh along the z(axis=2) axis
#     # chi_i: where the light ray starts at this time step
#     start_index = jnp.argmax(conf.mesh_chi>=chi_i) # will integrate the density fields that >= chi(a_start)
#     return dynamic_slice_in_dim(grad_mesh, start_index, slice_size=conf.lens_mesh_shape[2], axis=2)
