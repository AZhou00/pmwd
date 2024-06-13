from functools import partial

import jax.numpy as jnp
import jax
from jax import custom_vjp, linearize, jvp, vjp, jit
from pmwd.scatter import scatter, scatter_ray
from pmwd.gather import gather_ray
from pmwd.pm_util import fftfreq, fftfwd, fftinv
from pmwd.boltzmann import chi_a, r_a, r_chi, growth, growth_chi, AD_a, a_chi
from pmwd.ray_mesh import compute_ray_mesh, ray_mesh_center
from pmwd.sto.so import sotheta, pot_sharp, grad_sharp

def deconv_tophat(kvec, width, field):
    """perform 2D sinc^2 deconv in Fourier space. kvec from sparse meshgrid."""
    kx = kvec[0]
    ky = kvec[1]
    kernel = jnp.sinc(kx * width / 2 / jnp.pi) ** 2
    kernel *= jnp.sinc(ky * width / 2 / jnp.pi) ** 2
    field = jnp.where(kernel != 0, field / kernel, 0)
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


def grad_phi(a, ptcl, cosmo, conf):
    """
    Compute the 3D gradient of the potential field at 3D mesh points.
    """
    # length 3 tuple, ith has shape (mesh_shape[i], 1, 1)
    kvec_ptc = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype) # unit of [1/Mpc]

    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity
    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype) 

    # Solve Poisson's equation for the potential
    # fft mesh_shape, normalization canceled by that of irfftn below
    dens = fftfwd(dens)  
    pot = laplace(kvec_ptc, dens, cosmo) # apply -1/k^2 * (...)

    if conf.so_type is not None:  # spatial optimization
        theta = sotheta(cosmo, conf, a)
        pot = pot_sharp(pot, kvec_ptc, theta, cosmo, conf, a)

    # Compute the negative gradient of the potential, apply -ik * (...)
    grad_phi3D = []
    for k in kvec_ptc:
        grad_tmp = neg_grad(k, pot, conf.cell_size)
        if conf.so_type is not None:  # spatial optimization
            grad_tmp = grad_sharp(grad_tmp, k, theta, cosmo, conf, a)
        grad_tmp = fftinv(grad_tmp, shape=conf.mesh_shape)
        grad_tmp = grad_tmp.astype(conf.float_dtype)
        # mesh_shape
        grad_phi3D.append(grad_tmp)
    # (mesh_shape + (3,))
    grad_phi3D = jnp.stack(grad_phi3D, axis=-1) 
    return grad_phi3D


def rotation_matrix_x(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@jit
def rotate_coordinates(coords, angles):
    """
    Rotate coordinates based on a vector of angles [theta_x, theta_y, theta_z].
    """
    theta_x, theta_y, theta_z = angles
    Rx = rotation_matrix_x(theta_x)
    Ry = rotation_matrix_y(theta_y)
    Rz = rotation_matrix_z(theta_z)

    R = Rz @ Ry @ Rx

    return jnp.dot(coords, R.T)


@jit
def shift_z_conditional(coord3D, delta_z, threshold):
    """Shift the z-coordinate by delta_z for all z-coordinates less than the threshold.
    Args:
    coord3D: A 4D array of shape (Nx, Ny, Nz, 3), containing x, y, z coordinates.
    """
    z_coords = coord3D[..., 2]
    new_z_coords = jnp.where(z_coords+delta_z < threshold, z_coords + delta_z, z_coords)
    new_coord3D = coord3D.at[..., 2].set(new_z_coords)
    return new_coord3D


def mesh3D_coord(chi_i, chi_f, cosmo, conf, periodic=False):
    """
    assumes chi_f - chi_i < conf.box_size[2]
    i.e., n_f - n_i = 0 or 1
    conf.dim == 3
    x: x coordinates of the 3D mesh, shape (N_x,), x.mean() = 0, unit [L]
    y: y coordinates of the 3D mesh, shape (N_y,), y.mean() = 0, unit [L]
    r: radial comoving distance, shape (N_z,), unit [L]
    """
    # monolit. mesh
    x = jnp.arange(conf.mesh_shape[0]).astype(conf.float_dtype) * conf.cell_size
    y = jnp.arange(conf.mesh_shape[1]).astype(conf.float_dtype) * conf.cell_size
    z = jnp.arange(conf.mesh_shape[2]).astype(conf.float_dtype) * conf.cell_size
    x_mean, y_mean = x.mean(), y.mean()
    x, y = x - x_mean, y - y_mean
    r = r_chi(z, cosmo, conf).astype(conf.float_dtype)
    # tuple of 3, each has shape (N_x, N_y, N_z)
    coord3D = jnp.meshgrid(*[x, y, r], indexing="ij")
    coord3D = jnp.stack(coord3D, axis=-1) # shape (N_x, N_y, N_z, 3)
    coord3D = coord3D.reshape(-1, conf.dim) # shape (N_x * N_y * N_z, 3)

    return coord3D, z, r


def defl_pbc(chi_i, chi_f, cosmo, conf, pot, D_c, a_c, ray_cell_size, periodic=True):
    """
    assumes chi_f - chi_i < conf.box_size[2]
    i.e., n_f - n_i = 0 or 1
    conf.dim == 3
    x: x coordinates of the 3D mesh, shape (N_x,), x.mean() = 0, unit [L]
    y: y coordinates of the 3D mesh, shape (N_y,), y.mean() = 0, unit [L]
    r: radial comoving distance, shape (N_z,), unit [L]
    """
    x = jnp.arange(conf.mesh_shape[0]).astype(conf.float_dtype) * conf.cell_size
    y = jnp.arange(conf.mesh_shape[1]).astype(conf.float_dtype) * conf.cell_size
    z = jnp.arange(conf.mesh_shape[2]).astype(conf.float_dtype) * conf.cell_size
    x_mean, y_mean,z_mean = x.mean(), y.mean(), z.mean()
    x, y, z = x - x_mean, y - y_mean, z - z_mean
    coord3D = jnp.meshgrid(*[x, y, z], indexing="ij")
    coord3D = jnp.stack(coord3D, axis=-1) # shape (N_x, N_y, N_z, 3)
    coord3D = coord3D.reshape(-1, conf.dim) # shape (N_x * N_y * N_z, 3)

    # apply PBC
    n_i = int(chi_i // conf.box_size[2])
    n_f = int(chi_f // conf.box_size[2])

    # only works for square boxes
    # generate 50 floats using rngkey(0)
    rng_x = jax.random.PRNGKey(int(chi_i)+1)
    rng_y = jax.random.PRNGKey(int(chi_i)+2)
    rng_z = jax.random.PRNGKey(int(chi_i)+3)
    rng_trans = jax.random.PRNGKey(int(chi_i)+4)
    rng_trans_z = jax.random.PRNGKey(int(chi_i)+5)

    rand_trans = jax.random.uniform(rng_trans, shape=(50,), dtype=conf.float_dtype)
    rand_trans_z = jax.random.uniform(rng_trans_z, shape=(50,), dtype=conf.float_dtype)
    rand_angx = jax.random.randint(rng_x, shape=(50,), minval=0, maxval=4, dtype=jnp.int32)
    rand_angy = jax.random.randint(rng_y, shape=(50,), minval=0, maxval=4, dtype=jnp.int32)
    rand_angz = jax.random.randint(rng_z, shape=(50,), minval=0, maxval=4, dtype=jnp.int32)
    rand_angx = jnp.asarray(rand_angx).astype(conf.float_dtype)
    rand_angy = jnp.asarray(rand_angy).astype(conf.float_dtype)
    rand_angz = jnp.asarray(rand_angz).astype(conf.float_dtype)

    # rotate
    angle_x = jnp.pi * rand_angx[0] / 2
    angle_y = jnp.pi * rand_angy[1] / 2
    angle_z = jnp.pi * rand_angz[2] / 2
    coord3D = rotate_coordinates(coord3D, angles=jnp.array([angle_x, angle_y, angle_z]))

    # shift
    delta = jnp.array([
        rand_trans[n_i] * conf.box_size[0],
        rand_trans[-n_i-1] * conf.box_size[1],
        rand_trans_z[n_i] * conf.box_size[2]])
    coord3D = coord3D + delta
    # boundary condition
    coord3D = coord3D % jnp.array(conf.box_size).astype(conf.float_dtype)
    # shift to center on x/y axis, and shift z
    delta = jnp.array([-x_mean, -y_mean, n_i * conf.box_size[2]])
    coord3D = coord3D + delta
    coord3D = shift_z_conditional(coord3D, conf.box_size[2], chi_f)

    pot = pot.reshape(-1, 3)
    pot = rotate_coordinates(pot, angles=jnp.array([angle_x, angle_y, angle_z]))

    # apply kernel
    pot = pot*jnp.where((coord3D[..., 2] >= chi_i) & (coord3D[..., 2] <= chi_f), 2*r_chi(coord3D[..., 2], cosmo, conf)/conf.c, 0.0)[:,None]
    pot = pot * jnp.where((coord3D[..., 2] >= conf.chi_rt_mincut), 1.0, 0.0)[:, None]
    pot = pot * growth_chi(coord3D[..., 2], cosmo, conf)[:, None] / D_c
    pot = pot * a_c / a_chi(coord3D[..., 2], cosmo, conf)[:, None]  # growth correction
    pot = pot / a_c  # Poisson factor
    pot = pot * jnp.where(r_chi(coord3D[..., 2], cosmo, conf) > 1e-3, conf.ptcl_cell_vol / ray_cell_size**2 / r_chi(coord3D[..., 2], cosmo, conf)**2, 0.0)[:,None]
    pot = pot.astype(conf.float_dtype)
    pot = pot[..., 0:2] # transverse gradient

    return coord3D, pot

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
    coordr = coord3D[:, 2].astype(conf.float_dtype) # (N_x * N_y * N_z,)
    coord2D = coord3D[:, 0:2].astype(conf.float_dtype) # (N_x * N_y * N_z, 2)
    coord2D /= (coordr + 1e-5)[:, jnp.newaxis]  # (N_x * N_y * N_z, 2)
    val_mesh2D = scatter_ray(
        disp=coord2D,
        val=val_mesh3D,
        conf=conf,
        offset=ray_mesh_center(mesh2D_cell_size, mesh2D_mesh_shape, dtype=conf.float_dtype),
        cell_size=mesh2D_cell_size,
        mesh_shape=mesh2D_mesh_shape,
    )
    return val_mesh2D


def deflection_field(ptcl, a_i, a_f, a_c, grad_phi3D, ray_cell_size, ray_mesh_shape, cosmo, conf):
    """
    Twirl of rays on the image plane
    a_i: scale factor where the ray tracing begins
    a_f: scale factor where the ray tracing ends
    a_c: any scale factor between ai and af, to make the calculation more accurate.
    grad_phi3D: 3D gradient of the potential field at 3D mesh points, shape (N_x, N_y, N_z, 3)

    All calculations follow unit in L=[Mpc] and T=[1/H0]
    """

    chi_i = chi_a(a_i, cosmo, conf)
    chi_f = chi_a(a_f, cosmo, conf)
    D_c = growth(a_c, cosmo, conf, order=1, deriv=0)

    # print('------------------')
    # print('compute lensing maps')
    # print('------------------')
    # print(f"{'a_i':>15} = {a_i}, {'a_f':>15} = {a_f}")
    # print(f"{'chi_i':>15} = {chi_i}, {'chi_f':>15} = {chi_f}")
    # print number of particle mesh planes
    # print(f"{'# planes covered':>15} = {(chi_f-chi_i)/conf.cell_size}")
    # particle mesh ------------------
    
    # uncomment for non-pbc
    # coord3D, z, r = mesh3D_coord(chi_i, chi_f, cosmo, conf, periodic=True)
    
    # Potential gradient ------------------
    # (mesh_shape + (3,))
    # grad_phi3D /= (conf.mesh_size / conf.ptcl_num) # mass scale correction
    
    # uncomment for non-pbc
    # # compute lensing kernel on the z axis and broadcast to x & y
    # kernel = jnp.where((z >= chi_i) & (z <= chi_f), 2 * r / conf.c, 0.0)  # shape (N_z,)
    # kernel *= jnp.where((z >= conf.chi_rt_mincut), 1.0, 0.0)
    # kernel *= growth_chi(z, cosmo, conf) / D_c
    # kernel /= a_c # Poisson factor
    # kernel *= jnp.where(r > 1e-3, conf.ptcl_cell_vol / ray_cell_size**2 / r**2, 0.0)
    # kernel = kernel.astype(conf.float_dtype)
    # grad_phi3D = grad_phi3D[..., 0:2] # transverse gradient
    # defl_mesh3D = jnp.einsum("xyzv,z->xyzv", grad_phi3D, kernel) # grad_phi3D shape = (N_x, N_y, N_z, 2)
    # defl_mesh3D = defl_mesh3D.reshape(-1, 2)  # (mesh_shape + (2,)) -> (mesh_size, 2)
    
    coord3D, defl_mesh3D = defl_pbc(chi_i, chi_f, cosmo, conf, grad_phi3D, D_c, a_c, ray_cell_size)
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
    # smoothing
    smoothing_width = ray_cell_size/conf.ray_mesh_iota
    defl_2D_fft_x = smoothing_gaussian(kvec_ray, smoothing_width, defl_2D_fft_x)
    defl_2D_fft_y = smoothing_gaussian(kvec_ray, smoothing_width, defl_2D_fft_y)
    defl_2D_x = fftinv(defl_2D_fft_x, shape=ray_mesh_shape)
    defl_2D_y = fftinv(defl_2D_fft_y, shape=ray_mesh_shape)
    defl_2D_smth = jnp.stack([defl_2D_x, defl_2D_y], axis=-1)
    # shape (ray_mesh_shape[0], ray_mesh_shape[1], 2)
    return defl_2D_smth


def lensing(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf):
    """
    1. Compute mesh
    2. Compute deflection field
    3. Compute incremental ray states
    """
    r_i = r_a(a_i, cosmo, conf)
    r_f = r_a(a_f, cosmo, conf)
    ray_cell_size, ray_mesh_shape = compute_ray_mesh(r_i, r_f, conf)
    offset = ray_mesh_center(ray_cell_size, ray_mesh_shape, dtype=conf.float_dtype)
    # (ray_mesh_shape[0], ray_mesh_shape[1], 2)
    defl_2D = deflection_field(ptcl, a_i, a_f, a_c, grad_phi3D, ray_cell_size, ray_mesh_shape, cosmo, conf)

    f = partial(
        gather_ray,
        conf=conf,
        mesh=defl_2D,
        offset=offset,
        cell_size=ray_cell_size,
        wrap=False,
    )

    # first/second column of the distortion matrix, (ray_num, 2)
    Ax = ray.A[:,:,0].reshape(-1,2)
    Ay = ray.A[:,:,1].reshape(-1,2)
    # ray positions, shape = (ray_num, 2)
    primals = ray.pos_ip(dtype=conf.float_dtype)
    # useful notes on linearize https://github.com/google/jax/issues/526
    deta, f_jvp = linearize(f, primals)
    dB_x = f_jvp(Ax) # (ray_num, 2)
    dB_y = f_jvp(Ay) # (ray_num, 2)
    dB = jnp.stack([dB_x, dB_y], axis=-1)
    return deta, dB
