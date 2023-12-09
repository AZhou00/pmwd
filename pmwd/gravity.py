import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import dynamic_slice_in_dim
from pmwd.scatter import scatter,_scatter_rt
from pmwd.gather import gather
from pmwd.pm_util import fftfreq, fftfwd, fftinv
from pmwd.boltzmann import distance

@custom_vjp
def laplace(kvec, src, cosmo=None):
    """Laplace kernel in Fourier space."""
    k2 = sum(k**2 for k in kvec)

    pot = jnp.where(k2 != 0, - src / k2, 0)

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


def gravity(a, ptcl, cosmo, conf):
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""
    print('fnc: gravity')

    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)

    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = fftfwd(dens)  # normalization canceled by that of irfftn below

    pot = laplace(kvec, dens, cosmo) # mesh shape in Fourier space

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)
        grad = fftinv(grad, shape=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32

        grad = gather(ptcl, conf, grad)

        acc.append(grad)

    acc = jnp.stack(acc, axis=-1)

    return acc

# -------------------------------------------------------------------------- #
# Computing 2d gradient for lensing 
# -------------------------------------------------------------------------- #
def lensing(a_i, a_f, ptcl, cosmo, conf):
    """
    swirl of light rays on the 2d image plane
    ai: scale factor where the ray tracing begins
    af: scale factor where the ray tracing ends
    """
    print('fnc: lensing')

    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)

    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = fftfwd(dens)  # normalization canceled by that of irfftn below

    pot = laplace(kvec, dens, cosmo) # mesh shape in Fourier space
    print('pot.shape', pot.shape)

    grad_mesh = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)

        grad = fftinv(grad, shape=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32, mesh_shape

        grad_mesh.append(grad)

    grad_mesh = jnp.stack(grad_mesh, axis=-1) # mesh_shape + (3,)
    print(grad_mesh.shape) 

    print('scattering to 2d grid using _scatter')
    chi_i = distance(a_i, cosmo, conf)
    chi_f = distance(a_f, cosmo, conf)

    # query the lens plane
    grad_mesh = slice(chi_i,grad_mesh,cosmo,conf)  # lens_mesh_shape + (3,)
    assert grad_mesh.shape == conf.lens_mesh_shape, f"grad_mesh.shape={grad_mesh.shape}, conf.lens_mesh_shape={conf.lens_mesh_shape}"

    # define lens plane mesh coordinates, the particle mesh is defined in regular comoving distance grid
    x = jnp.arange(conf.lens_mesh_shape[0])*conf.cell_size
    y = jnp.arange(conf.lens_mesh_shape[1])*conf.cell_size
    z = jnp.arange(conf.lens_mesh_shape[2])*conf.cell_size + chi_i 
    
    # mimicking how the particle grid is set up
    coords = jnp.meshgrid(*[x,y,z], indexing='ij') 
    coords = jnp.stack(coords, axis=-1).reshape(-1, conf.dim)  # shape (conf.lens_mesh_size, 3)
    coord_z = coords[:,2] # shape (conf.lens_mesh_size,)

    # compute the 2d image plane coordiante of each 3d mesh point
    coords -= jnp.array([conf.obsv_origin_cmv[0], conf.obsv_origin_cmv[1], 0])
    coords /= coord_z
    coords = coords[:,0:2] # drop the z coords, shape (conf.lens_mesh_size, 2)

    # compute lens kernel as a function radial comoving distance
    # TODO: add growth factor correction
    # TODO: add r(chi) function, maybe this can be cached
    rchi = coord_z
    lens_kernel = jnp.where((coord_z>=chi_i) & (coord_z<=chi_f), rchi, jnp.zeros(coord_z)) # (conf.lens_mesh_size,)

    # apply the lens kernel
    grad_mesh = grad_mesh.reshape(-1, 3) # (lens_mesh_size, 3)
    grad_mesh = jnp.einsum('ij,i->ij', grad_mesh, lens_kernel) # (lens_mesh_size, 3)

    grad_2d = _scatter_rt(
        pmid=jnp.zeros_like(coords),
        disp=coords, 
        conf=conf, # _scatter_rt is _scatter without conf.chunk_size dependence
        mesh=conf.ray_mesh_shape, 
        val=grad_mesh, 
        offset=None, 
        cell_size=conf.cell_size)

    return grad_2d

def slice(chi_i,grad_mesh,cosmo,conf):
    # slice the z(axis=2) range of the mesh
    # chi_i: where the light ray starts at this time step
    start_index = jnp.argmax(conf.mesh_chi>=chi_i) # will integrate the density fields that >= chi(a_start)
    return dynamic_slice_in_dim(grad_mesh, start_index, slice_size=conf.lens_mesh_shape[2], axis=2)