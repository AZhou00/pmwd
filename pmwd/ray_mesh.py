import jax.numpy as jnp
import jax 
from functools import reduce

def fftlen(n, platform=None):
    """Find the next fast length for FFT on the platform.

    Parameters
    ----------
    n : float
        FFT length to start searching from.
    platform : str or xla_client.Client, optional
        Platform supported by XLA. Default is the platform of the default backend.

    .. _ducc0/fft good_size_real:
        https://gitlab.mpcdf.mpg.de/mtr/ducc/-/blob/ducc0/src/ducc0/fft/fft.h
    .. _rocFFT:
        https://rocm.docs.amd.com/projects/rocFFT
    .. _cuFFT:
        https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance

    """
    platform = jax.lib.xla_bridge.get_backend(platform=platform).platform

    if platform == 'cpu':
        radices = [2, 3, 5]
    elif platform == 'rocm':
        radices = [2, 3, 5, 7, 11, 13, 17]
    elif platform in ['cuda', 'gpu']:
        radices = [2, 3] #[2, 3, 5, 7]
    else:
        raise NotImplementedError(f'Platform {platform} is not supported.')

    return fftlen_(n, radices)


def fftlen_(n, radices):
    """Find the next fast length for FFT given radices.

    Parameters
    ----------
    n : float
        FFT length to start searching from.
    radices : sequence of (prime) int
        FFT radices.

    """
    if n <= max(radices):
        return n

    #FIXME if need ensure_compile_time_eval
    # log_radix_2n = jnp.log(2*n)/jnp.log(radix) # log(2*n, radix)
    powers = [radix ** (jnp.arange(1 + jnp.floor(jnp.log(2*n)/jnp.log(radix)))) for radix in radices]
    products = reduce(jnp.kron, powers)
    return products[products >= n].min().item()

def compute_ray_mesh(mu_2D, M_2D_x, M_2D_y, r_l, r_u, l_3D, iota=1, p_x=0, p_y=0):
    """
    Compute the size and shape of the ray mesh for a given lens plane and source plane.

    Args:
    mu_2D (float): distance between rays (i.e., ray grid spacing) at $z=0$, [rad]
    M_2D_x (int): how many rays on side x
    M_2D_y (int): how many rays on side y; ray grid has $M_{2D_x} \times M_{2D_y}$ rays
    r_l (float): lower bound of the lens plane's radial comoving distance, [Mpc]
    r_u (float): upper bound of the lens plane's radial comoving distance, [Mpc]
    l_3D (float): 3D $\grad \npot$ mesh cell size in comoving space, [Mpc]
    iota (float): The precision parameter; $\epsilon \in (0.0,1]$, the lower the better; typically 1/2.
    p_x (int): min number of mesh point padding on side x
    p_y (int): min number of mesh point padding on side y
        padding defined by
        $$ N_{2D} \nu_{2D} \leq M_{2D} \mu_{2D} + p_{min} \lambda_{lim} $$
    Returns:
    nu_2D (float): ray mesh spacing
    N_2D_x (int): the number of mesh points along x
    N_2D_y (int): the number of mesh points along y
    """
    if p_x == 0:
        p_x = 5
    if p_y == 0:
        p_y = 5
        
    r_mean = (r_l + r_u) / 2.0
    lambda_lim = jnp.max(jnp.array([l_3D / r_mean, mu_2D]))
    
    nu_2D = iota * lambda_lim
    Nx = M_2D_x*mu_2D/nu_2D+p_x/iota
    Nx = jnp.array(fftlen(Nx, platform=None))
    Nx = Nx.astype(jnp.int32)
    Ny = M_2D_y*mu_2D/nu_2D+p_y/iota
    Ny = jnp.array(fftlen(Ny, platform=None))
    Ny = Ny.astype(jnp.int32)
    return nu_2D, Nx, Ny

def ray_mesh_diagnostic(mu_2D, M_2D_x, M_2D_y, r_l, r_u, l_3D, iota=0.5, p_x=0, p_y=0):
    nu_2D, N_2D_x, N_2D_y = compute_ray_mesh(mu_2D, M_2D_x, M_2D_y, r_l, r_u, l_3D, iota, p_x, p_y)

    # print aligned table
    # r_mean = 2.0 / (1.0 / r_l + 1.0 / r_u)
    r_mean = (r_l + r_u) / 2.0
    
    print('------------------')
    # particle mesh reso: l_3D / r_mean
    print(f"{'ptcl mesh res':>15} = {l_3D/r_mean*180*60/jnp.pi:.2e} [arcmin]")
    # ray mesh reso: mu_2D
    print(f"{'mu_2D':>15} = {mu_2D*180*60/jnp.pi:.2e} [arcmin]")
    # limiting resolution
    lambda_lim = jnp.max(jnp.array([l_3D / r_mean, mu_2D]))
    print(f"{'res lim':>15} = {lambda_lim*180*60/jnp.pi:.2e} [arcmin]")

    # printy mesh grid
    print(f"{'nu_2D':>15} = {nu_2D*180*60/jnp.pi:.2e} [arcmin]")
    # print log2 mesh grid
    print(f"{'log2_N_2D_x':>15} = {jnp.log2(N_2D_x)}, {'log2_N_2D_y':>15} = {jnp.log2(N_2D_y)}")
    print(f"{'N_2D_x':>15} = {N_2D_x}, {'N_2D_y':>15} = {N_2D_y}")
    return


# def compute_ray_mesh_old(mu_2D, M_2D_x, M_2D_y, r_l, r_u, l_3D, eps=1, p_x=0, p_y=0):
#     """
#     Compute the size and shape of the ray mesh for a given lens plane and source plane.

#     Args:
#     mu_2D (float): distance between rays (i.e., ray grid spacing) at $z=0$, [rad]
#     M_2D_x (int): how many rays on side x
#     M_2D_y (int): how many rays on side y; ray grid has $M_{2D_x} \times M_{2D_y}$ rays
#     r_l (float): lower bound of the lens plane's radial comoving distance, [Mpc]
#     r_u (float): upper bound of the lens plane's radial comoving distance, [Mpc]
#     l_3D (float): 3D $\grad \npot$ mesh cell size in comoving space, [Mpc]
#     eps (float): The precision parameter; $\epsilon \in (0.0,1]$, the lower the better; typically 1/2.
#     p_x (int): number of mesh point padding on side x
#     p_y (int): number of mesh point padding on side y
#         padding defined by
#         $$ N_{2D} \nu_{2D} = M_{2D} \mu_{2D} + p_{2D} \lambda_{lim} $$
#     Returns:
#     nu_2D (float): ray mesh spacing
#     N_2D_x (int): the number of mesh points along x
#     N_2D_y (int): the number of mesh points along y
#     """
#     if p_x == 0:
#         p_x = 5
#     if p_y == 0:
#         p_y = 5
    
#     # harmonic mean of the lens plane's comoving radial distance
#     # r_mean = 2.0 / (1.0 / r_l + 1.0 / r_u)
#     r_mean = (r_l + r_u) / 2.0
#     lambda_lim = jnp.max(jnp.array([l_3D / r_mean, mu_2D]))

#     # ray mesh spacing
#     nu_2D = eps * lambda_lim

#     # the lower bound of the number of mesh points along x
#     N_2D_x_lb = 1 / eps * (M_2D_x * mu_2D / lambda_lim + p_x)
#     log2_N_2D_x = jnp.ceil(jnp.log2(N_2D_x_lb))
#     # log2_N_2D_x = jnp.array(log2_N_2D_x, dtype=jnp.int32)
#     log2_N_2D_x = log2_N_2D_x.astype(jnp.int32)
#     # the number of mesh points along x
#     N_2D_x = 2**log2_N_2D_x

#     # the lower bound of the number of mesh points along y
#     N_2D_y_lb = 1 / eps * (M_2D_y * mu_2D / lambda_lim + p_y)
#     log2_N_2D_y = jnp.ceil(jnp.log2(N_2D_y_lb))
#     # log2_N_2D_y = jnp.array(log2_N_2D_y, dtype=jnp.int32)
#     log2_N_2D_y = log2_N_2D_y.astype(jnp.int32)
#     # the number of mesh points along y
#     N_2D_y = 2**log2_N_2D_y
    
#     # # the lower bound of the number of mesh points along x
#     # N_2D_x_lb = M_2D_x*jnp.min(jnp.array([mu_2D*r_mean/l_3D, 1])) + p_x
#     # N_2D_x_lb /= eps
#     # log2_N_2D_x = jnp.ceil(jnp.log2(N_2D_x_lb))
#     # log2_N_2D_x = log2_N_2D_x.astype(jnp.int32)
#     # # the number of mesh points along x
#     # N_2D_x = 2**log2_N_2D_x

#     # # the lower bound of the number of mesh points along y
#     # N_2D_y_lb = M_2D_y*jnp.min(jnp.array([mu_2D*r_mean/l_3D, 1])) + p_y
#     # N_2D_y_lb /= eps
#     # log2_N_2D_y = jnp.ceil(jnp.log2(N_2D_y_lb))
#     # log2_N_2D_y = log2_N_2D_y.astype(jnp.int32)
#     # # the number of mesh points along y
#     # N_2D_y = 2**log2_N_2D_y
#     # nu_2D = M_2D_y/N_2D_y*mu_2D+p_y*lambda_lim/N_2D_y
    
#     return nu_2D, N_2D_x, N_2D_y

# def compute_ray_mesh_max(mu_2D, M_2D_x, M_2D_y, eps=0.5, p_x=0, p_y=0):
#     """
#     compute_ray_mesh in lim r -> infinity <=> lambda_lim = mu_2D
#     """
#     if p_x == 0:
#         p_x = 20
#     if p_y == 0:
#         p_y = 20

#     # ray mesh spacing
#     nu_2D = eps * mu_2D

#     # the lower bound of the number of mesh points along x
#     N_2D_x_lb = 1 / eps * (M_2D_x + p_x)
#     log2_N_2D_x = jnp.ceil(jnp.log2(N_2D_x_lb))
#     log2_N_2D_x = jnp.asarray(log2_N_2D_x, dtype=jnp.int32)
#     # the number of mesh points along x
#     N_2D_x = 2**log2_N_2D_x

#     # the lower bound of the number of mesh points along y
#     N_2D_y_lb = 1 / eps * (M_2D_y + p_y)
#     log2_N_2D_y = jnp.ceil(jnp.log2(N_2D_y_lb))
#     log2_N_2D_y = jnp.asarray(log2_N_2D_y, dtype=jnp.int32)
#     # the number of mesh points along y
#     N_2D_y = 2**log2_N_2D_y

#     return nu_2D, N_2D_x, N_2D_y