import jax.numpy as jnp
from jax import custom_vjp,linearize,jvp,vjp,jit
from jax.lax import dynamic_slice_in_dim
from pmwd.scatter import scatter,_scatter_rt
from pmwd.gather import gather, _gather_rt
from pmwd.pm_util import fftfreq, fftfwd, fftinv
from pmwd.boltzmann import chi_a, r_a, r_chi, growth, growth_chi
from pmwd.ray_mesh import compute_ray_mesh, ray_mesh_center

from functools import partial

import matplotlib.pyplot as plt
import matplotlib
from vermeer import snapshot  # .pm.snapshot import


@custom_vjp
def laplace(kvec, src, cosmo=None):
    """Laplace kernel in Fourier space."""
    k2 = sum(k**2 for k in kvec)

    pot = jnp.where(k2 != 0, -src / k2, 0)

    return pot


def laplace_fwd(kvec, src, cosmo):
    pot = laplace(kvec, src, cosmo)
    return pot, (kvec, cosmo)


def laplace_bwd(res, pot_cot):
    """Custom vjp to avoid NaN when using where, as well as to save memory.

    .. _JAX FAQ:
        https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    """
    kvec, cosmo = res
    src_cot = laplace(kvec, pot_cot, cosmo)
    return None, src_cot, None


laplace.defvjp(laplace_fwd, laplace_bwd)


def neg_grad(k, pot, spacing):
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = neg_ik * pot

    return grad


def grad_phi(ptcl, cosmo, conf):
    """
    Compute the 3D gradient of the potential field at 3D mesh points.
    """
    # length 3 tuple, ith has shape (mesh_shape[i], 1, 1)
    kvec_ptc = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype) # unit of [1/Mpc]
    # print('kvec', kvec.dtype)
    # RHS of Poisson's equation
    dens = scatter(ptcl, conf) # mesh_shape
    dens -= 1  # overdensity
    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype) 
    dens *= (cosmo.h.astype(conf.float_dtype))**2 # H0^2 term
    dens /= (conf.mesh_size / conf.ptcl_num) # mass scale correction
    # dens /= a_c 
    # dens *= a_c**2 

    # # potential is evaluated at the middle of the time step
    # Solve Poisson's equation for the potential 
    dens = fftfwd(dens)  # normalization canceled by that of irfftn below
    # print('dens.shape', dens.shape) # fft mesh_shape
    pot = laplace(kvec_ptc, dens, cosmo) # apply -1/k^2 * (...)
    # print('pot.shape', pot.shape) 

    # Compute the negative gradient of the potential, apply -ik * (...)
    grad_phi3D = []
    for k in kvec_ptc:
        grad_tmp = neg_grad(k, pot, conf.cell_size)
        grad_tmp = fftinv(grad_tmp, shape=conf.mesh_shape)
        grad_tmp = grad_tmp.astype(conf.float_dtype)
        # mesh_shape
        grad_phi3D.append(grad_tmp)
    # (mesh_shape + (3,))
    grad_phi3D = jnp.stack(grad_phi3D, axis=-1) 
    return grad_phi3D


def deconv_tophat(kvec, width, field):
    """perform 2D sinc^2 deconv in Fourier space. kvec from sparse meshgrid."""
    kx = kvec[0]
    ky = kvec[1]
    kernel = jnp.sinc(kx * width / 2 / jnp.pi) ** 2
    kernel *= jnp.sinc(ky * width / 2 / jnp.pi) ** 2
    field = jnp.where(kernel != 0, field / kernel, 0)
    return field


def smoothing_gaussian(kvec, width, field):
    """perform 2D Gaussian smoothing in Fourier space.
    FT(Gaussian) = exp(-0.5 * (kx^2 + ky^2) * width^2)
    Args:
        kvec: kvec from sparse meshgrid.
        width: width of the Gaussian kernel in real space.
        field: field to be smoothed.
    """
    kx = kvec[0]
    ky = kvec[1]
    kernel = jnp.exp(-0.5 * (kx**2 + ky**2) * width**2)
    field = field * kernel
    return field

def smoothing_tophat(kvec, width, field):
    """perform 2D tophat smoothing in Fourier space.
    Args:
        kvec: kvec from sparse meshgrid.
        width: width of the tophat kernel in real space.
        field: field to be smoothed.
    """
    kx = kvec[0]
    ky = kvec[1]
    kernel = jnp.sinc(kx * width / 2 / jnp.pi)
    kernel *= jnp.sinc(ky * width / 2 / jnp.pi)
    field = field * kernel
    return field

def project(
    val_mesh3D,
    coord3D,
    mesh2D_cell_size,
    mesh2D_mesh_shape,
    conf,
):
    """
    Project points on a 3D mesh down to a 2D mesh on the image plane with small angle approximation.

    Args:
        val_mesh3D: 3D mesh to be projected, shape (N_x * N_y * N_z, N_v)
        coord3D: 3D comoving coordinates of the 3D mesh, shape (N_x * N_y * N_z, 3)
        conf: Configuration file,
    Returns:
        val_mesh2D: projected 2D mesh, shape (mesh2D_mesh_shape + (N_v,))
    """
    mesh3D_size, N_v = val_mesh3D.shape

    # compute angular coordinates; the projected mesh is centered at x = y = 0

    # line of sight angular diameter distance
    # shape (N_x * N_y * N_z,)
    coordr = coord3D[:, 2].astype(conf.float_dtype)
    # drop the z coords, shape (N_x * N_y * N_z, 2)
    coord2D = coord3D[:, 0:2].astype(conf.float_dtype)
    
    # divide the first two coordinates by the radial comoving distance
    # should not use np.where and set coordr=0 entry to 0, because we need to remove these values
    coord2D /= (coordr + 1e-5)[:, jnp.newaxis]  # shape (N_x * N_y * N_z, 2)

    # mask out the 3D mesh points that falls considerably far away from the lens plane
    # note mesh2D here is already padded so do not need to pad any further
    mask = jnp.where(coord2D[:, 0] > mesh2D_mesh_shape[0] * mesh2D_cell_size / 2, 0, 1)
    mask *= jnp.where(
        coord2D[:, 0] < -mesh2D_mesh_shape[0] * mesh2D_cell_size / 2, 0, 1
    )
    mask *= jnp.where(coord2D[:, 1] > mesh2D_mesh_shape[1] * mesh2D_cell_size / 2, 0, 1)
    mask *= jnp.where(
        coord2D[:, 1] < -mesh2D_mesh_shape[1] * mesh2D_cell_size / 2, 0, 1
    )
    val_mesh3D = val_mesh3D * mask[:, jnp.newaxis]
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
    
    val_mesh2D = _scatter_rt(
        # this is place holder; positions of the 3D mesh points are given by 'disp';
        # shape (N_x * N_y * N_z, 2)
        pmid=jnp.zeros_like(coord2D).astype(conf.pmid_dtype),
        # positions of the 3D mesh points
        # shape (N_x * N_y * N_z, 2)
        disp=coord2D,
        conf=conf,
        # place holder for the 2D mesh
        mesh=jnp.zeros(mesh2D_mesh_shape + (N_v,), dtype=conf.float_dtype),
        # values of the 3D mesh points
        # shape (N_x * N_y * N_z, N_v)
        val=val_mesh3D, #
        # offset of the 2D mesh relative to the 3D mesh points
        offset=ray_mesh_center(mesh2D_cell_size, mesh2D_mesh_shape, dtype=conf.float_dtype),
        # cell size of the 2D mesh
        ray_cell_size=mesh2D_cell_size,
        ray_mesh_shape=mesh2D_mesh_shape,
        ray_mesh_size=mesh2D_mesh_shape[0] * mesh2D_mesh_shape[1],
        ray_num=conf.ray_num,
    )
    # shape (mesh2D_mesh_shape + (N_v,))
    return val_mesh2D

def mesh3D_coord(chi_i, chi_f, cosmo, conf):
    # assumes chi_f - chi_i < conf.box_size[2]
    # i.e., n_f - n_i = 0 or 1
    # conf.dim == 3
    # x: x coordinates of the 3D mesh, shape (N_x,), x.mean() = 0, unit [L]
    # y: y coordinates of the 3D mesh, shape (N_y,), y.mean() = 0, unit [L]
    # r: radial comoving distance, shape (N_z,), unit [L]

    b = conf.box_size[2]
    n_i = chi_i // b
    n_f = chi_f // b

    x = jnp.arange(conf.mesh_shape[0]).astype(conf.float_dtype) * conf.cell_size
    y = jnp.arange(conf.mesh_shape[1]).astype(conf.float_dtype) * conf.cell_size
    x, y = x - conf.box_size[0] / 2, y - conf.box_size[1] / 2
    
    z = jnp.arange(conf.mesh_shape[2],dtype=conf.float_dtype) * conf.cell_size
    z += n_i * b
    z_next_box = z + (n_f - n_i) * b
    z = jnp.where(z_next_box <= chi_f, z_next_box, z).astype(conf.float_dtype)
    r = r_chi(z, cosmo, conf).astype(conf.float_dtype)
    
    # tuple of 3, each has shape (N_x, N_y, N_z)
    coord3D = jnp.meshgrid(*[x, y, r], indexing="ij")
    # shape (N_x, N_y, N_z, 3)
    coord3D = jnp.stack(coord3D, axis=-1)
    # shape (N_x * N_y * N_z, 3)
    coord3D = coord3D.reshape(-1, conf.dim)
    return coord3D, z, r

def deflection_field(ptcl, a_i, a_f, a_c, grad_phi3D, ray_cell_size, ray_mesh_shape, cosmo, conf):
    """
    swirl of light rays on the 2d image plane
    a_i: scale factor where the ray tracing begins
    a_f: scale factor where the ray tracing ends
    a_c: any scale factor between ai and af, to make the calculation more accurate.

    all calculations follow unit in L=[Mpc] and T=[1/H0]
    """

    chi_i = chi_a(a_i, cosmo, conf)
    chi_f = chi_a(a_f, cosmo, conf)
    chi_maxsource = chi_a(conf.a_nbody_ray[-1], cosmo, conf)
    r_c = r_a(a_c, cosmo, conf)
    D_c = growth(a_c, cosmo, conf, order=1, deriv=0)

    # print('------------------')
    # print('compute lensing maps')
    # print('------------------')
    # print(f"{'a_i':>15} = {a_i}, {'a_f':>15} = {a_f}")
    # print(f"{'chi_i':>15} = {chi_i}, {'chi_f':>15} = {chi_f}")
    # print number of particle mesh planes
    # print(f"{'# planes covered':>15} = {(chi_f-chi_i)/conf.cell_size}")
    # particle mesh ------------------
    coord3D, z, r = mesh3D_coord(chi_i, chi_f, cosmo, conf)
    
    # Potential gradient ------------------
    # reuse grad_phi3D = grad_phi(a_c, ptcl, cosmo, conf)
    grad_phi3D /= a_c
    # (mesh_shape + (3,))

    # compute lensing kernel on the z axis and broadcast to the x&y axes
    kernel = jnp.where((z >= chi_i) & (z <= chi_f), 2 * r / conf.c, 0)  # shape (N_z,)
    kernel *= jnp.where(z > 0, 1, 0)
    kernel *= jnp.where((z >= conf.chi_rtmin) & (z <= chi_maxsource), 1, 0)
    kernel *= jnp.ones_like(r) * growth_chi(z, cosmo, conf) / D_c
    kernel *= jnp.where(r > 0, conf.ptcl_cell_vol / ray_cell_size**2 / r**2, 0)
    kernel = kernel.astype(conf.float_dtype)
    
    # grad_phi3D shape = (N_x, N_y, N_z, 3)
    # kernel shape = (N_z,)
    defl_mesh3D = jnp.einsum("xyzv,z->xyzv", grad_phi3D, kernel)
    # only need the x&y gradient
    defl_mesh3D = defl_mesh3D[..., 0:2]
    defl_mesh3D = defl_mesh3D.reshape(-1, 2)  # (mesh_shape + (2,)) -> (mesh_size, 2)
    defl_2D = project(
        val_mesh3D=defl_mesh3D,
        coord3D=coord3D,
        mesh2D_cell_size=ray_cell_size,
        mesh2D_mesh_shape=ray_mesh_shape,
        conf=conf,
    )
    
    # length 2 tuple, ith has shape (ray_mesh_shape[i], 1)
    kvec_ray = fftfreq(ray_mesh_shape, ray_cell_size, dtype=conf.float_dtype)
    defl_2D_fft_x = fftfwd(defl_2D[...,0])  # normalization canceled by that of irfftn below
    defl_2D_fft_y = fftfwd(defl_2D[...,1])  # normalization canceled by that of irfftn below
    # deconv scattering
    defl_2D_fft_x = deconv_tophat(kvec_ray, ray_cell_size, defl_2D_fft_x)
    defl_2D_fft_y = deconv_tophat(kvec_ray, ray_cell_size, defl_2D_fft_y)
    # deconv gathering
    defl_2D_fft_x = deconv_tophat(kvec_ray, ray_cell_size, defl_2D_fft_x)
    defl_2D_fft_y = deconv_tophat(kvec_ray, ray_cell_size, defl_2D_fft_y)
    # smooth with smoothing_tophat smoothing_gaussian
    smoothing_width = jnp.max(jnp.array([conf.cell_size / r_c, conf.ray_spacing])).astype(conf.float_dtype)
    # smoothing_width = ray_cell_size
    # smoothing_width /= jnp.sqrt(2)
    defl_2D_fft_x = smoothing_gaussian(kvec_ray, smoothing_width, defl_2D_fft_x)
    defl_2D_fft_y = smoothing_gaussian(kvec_ray, smoothing_width, defl_2D_fft_y)
    # defl_2D_fft_x = smoothing_tophat(kvec_ray, smoothing_width, defl_2D_fft_x)
    # defl_2D_fft_y = smoothing_tophat(kvec_ray, smoothing_width, defl_2D_fft_y)
    defl_2D_x = fftinv(defl_2D_fft_x, shape=ray_mesh_shape)
    defl_2D_y = fftinv(defl_2D_fft_y, shape=ray_mesh_shape)
    defl_2D_smth = jnp.stack([defl_2D_x, defl_2D_y], axis=-1)
    # if chi_i >2200:
    # visual_density_defl2D(ptcl, defl_2D_smth, coord3D, conf, z, chi_i, chi_f, ray_mesh_shape, ray_cell_size)
    return defl_2D_smth

def gather_eta(ray_pos_ip, ray_pmid, defl_2D_smth, ray_cell_size, conf):
    """Compute the kick operator on the ray poasitions.

    Args:
        ray_pos_ip (_type_): (ray_num, 2), ray.pos_ip()
        defl_2D_smth (_type_): (ray_mesh_shape[0], ray_mesh_shape[1], 2), smoothed deflection field
        ray_pmid (_type_): (ray_num, 2), ray.pmid
        ray_cell_size (float): cell size of the ray mesh in radians
        conf (_type_): _description_

    Returns:
        deta: $d\eta$ on the ray positions, shape (ray_num, 2)
    """
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
    return deta


def lensing(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf):
    
    r_i = r_a(a_i, cosmo, conf)
    r_f = r_a(a_f, cosmo, conf)
    ray_cell_size, ray_mesh_shape = compute_ray_mesh(r_i, r_f, conf)
    # (ray_mesh_shape[0], ray_mesh_shape[1], 2)
    defl_2D_smth = deflection_field(ptcl, a_i, a_f, a_c, grad_phi3D, ray_cell_size, ray_mesh_shape, cosmo, conf)
    f = partial(
        gather_eta,
        ray_pmid=ray.pmid,
        defl_2D_smth=defl_2D_smth,
        ray_cell_size=ray_cell_size,
        conf=conf,
    )
    
    # first/second column of the distortion matrix, (ray_num, 2)
    Ax = ray.A[:,:,0].reshape(-1,2)
    Ay = ray.A[:,:,1].reshape(-1,2)
    # ray positions, shape = (ray_num, 2)
    primals = ray.pos_ip()
    # useful notes on linearize https://github.com/google/jax/issues/526
    deta, f_jvp = linearize(f, primals)
    dB_x = f_jvp(Ax) # (ray_num, 2)
    dB_y = f_jvp(Ay) # (ray_num, 2)
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
