import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import dynamic_slice_in_dim
from pmwd.scatter import scatter,_scatter_rt
from pmwd.gather import gather, _gather_rt
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

    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype) # has unit

    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = fftfwd(dens)  # normalization canceled by that of irfftn below
    pot = laplace(kvec, dens, cosmo) # mesh shape in Fourier space

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size) # has unit
        grad = fftinv(grad, shape=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32

        grad = gather(ptcl, conf, grad)

        acc.append(grad)

    acc = jnp.stack(acc, axis=-1)

    return acc


def lensing(a_i, a_f, ptcl, ray, cosmo, conf):
    """
    swirl of light rays on the 2d image plane
    ai: scale factor where the ray tracing begins
    af: scale factor where the ray tracing ends
    """
    # print('fnc: lensing')

    # assert a_i <= 0.95
    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)

    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = fftfwd(dens)  # normalization canceled by that of irfftn below
    pot = laplace(kvec, dens, cosmo) # mesh shape in Fourier space
    # print('pot.shape', pot.shape)

    grad_mesh = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)

        grad = fftinv(grad, shape=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32, mesh_shape

        grad_mesh.append(grad)

    grad_mesh = jnp.stack(grad_mesh, axis=-1) # mesh_shape + (3,)
    # print(grad_mesh.shape) 

    # ------------------ lensing ------------------
    # print('scattering to 2d grid using _scatter_rt')
    chi_i = distance(a_i, cosmo, conf)
    chi_f = distance(a_f, cosmo, conf)

    # query the lens plane
    grad_mesh = slice(chi_i,grad_mesh,cosmo,conf)  # lens_mesh_shape + (3,)

    # define lens plane mesh coordinates, the particle mesh is defined in regular comoving distance grid
    x = jnp.arange(conf.lens_mesh_shape[0])*conf.cell_size
    y = jnp.arange(conf.lens_mesh_shape[1])*conf.cell_size
    z = jnp.arange(conf.lens_mesh_shape[2])*conf.cell_size + chi_i #TODO chi_i?
    
    # mimicking how the particle grid is set up
    coords = jnp.meshgrid(*[x,y,z], indexing='ij') 
    coords = jnp.stack(coords, axis=-1).reshape(-1, conf.dim)  # shape (conf.lens_mesh_size, 3)
    coord_z = coords[:,2] # shape (conf.lens_mesh_size,)
    # print('shape of coord_z', coord_z.shape)

    # compute the 2d image plane coordiante of each 3d mesh point
    coords -= jnp.array([conf.ray_origin[0], conf.ray_origin[1], 0])
    coords /= (coord_z+1e-2)[:,jnp.newaxis] # shape (conf.lens_mesh_size, 3) TODO: stability at coord_z=0
    # print(jnp.unique(coords))
    coords = coords[:,0:2] # drop the z coords, shape (lens_mesh_size, 2)
    # print(jnp.unique(coords))

    # compute lens kernel as a function radial comoving distance
    # TODO: add growth factor correction
    # TODO: add r(chi) function, maybe this can be cached
    rchi = coord_z
    # lensing kernel initialize at 0, then set.
    lens_kernel = jnp.where((coord_z>=chi_i) & (coord_z<=chi_f), rchi, jnp.zeros_like(coord_z)) # (conf.lens_mesh_size,)
    # scaling by cell
    lens_kernel *= jnp.where(rchi>0, conf.ptcl_cell_vol / conf.ray_spacing**2 / rchi**2, jnp.zeros_like(coord_z)) # (conf.lens_mesh_size,)
    # lens_kernel /= jnp.where(rchi>0, conf.ptcl_cell_vol / conf.ray_spacing**2 / rchi**2, jnp.ones_like(coord_z)*1e6) # (conf.lens_mesh_size,)
    # print(lens_kernel)
    
    # apply the lens kernel
    grad_mesh = grad_mesh.reshape(-1, 3) # (lens_mesh_size, 3)
    grad_mesh = jnp.einsum('ij,i->ij', grad_mesh, lens_kernel) # (lens_mesh_size, 3) #TODO: broadcast?
    # drop z axis
    grad_mesh = grad_mesh[:,0:2] # (lens_mesh_size, 2)
    # print('shape of grad_mesh', grad_mesh.shape) # (lens_mesh_size, 2)
    # print('shape of coords', coords.shape) # (lens_mesh_size, 2)
    # print('mesh_shape', conf.ray_mesh_shape + grad_mesh.shape[1:]) # ray_mesh_shape + (2,)

    offset = -jnp.array([conf.ray_mesh_fov[0],conf.ray_mesh_fov[1]])/2 #TODO: is this right?
    # _scatter_rt is _scatter without conf.chunk_size dependence
    # ray_mesh_shape + (2,)
    grad_2d = _scatter_rt(
        pmid=jnp.zeros_like(coords).astype(conf.pmid_dtype), # (lens_mesh_size, 2)
        disp=coords, # (lens_mesh_size, 2)
        conf=conf, 
        mesh=jnp.zeros(conf.ray_mesh_shape + grad_mesh.shape[1:], dtype=conf.float_dtype), # scatter to ray_mesh_shape 
        val = grad_mesh, # (lens_mesh_size, 2)
        offset=offset,
        cell_size=conf.ray_cell_size)
    # print('shape of grad_2d after scatter', grad_2d.shape)

    # _gather_rt is _gather without conf.chunk_size dependence
    # (ray_num, 2) 
    grad_2d = _gather_rt(
        pmid = jnp.zeros_like(ray.pmid).astype(conf.pmid_dtype), 
        disp = ray.pos_ip(), 
        conf = conf, 
        mesh = grad_2d, 
        val = jnp.zeros((conf.ray_num,2)), 
        offset = offset,
        cell_size = conf.ray_cell_size)
    # print('shape of grad_2d after gather', grad_2d.shape) 

    # this is actually *negative* grad 2d
    grad_2d *= 2 # GR
    # grad_2d *= conf.cell_size # d chi

    # TODO: scaling
    # scaling: total volume / volume per cell
    # rchi_f = chi_f
    # rchi_i = chi_i
    # transverse_distance_f = rchi_f * conf.ray_spacing
    # transverse_distance_i = rchi_i * conf.ray_spacing
    # # full pyramid - tip of the pyramid
    # total_volume = (1/3 * transverse_distance_f**2 * chi_f) - (1/3 * transverse_distance_i**2 * chi_i) 
    # scaling = total_volume / conf.ptcl_cell_vol
    # print(scaling)
    # grad_2d /= scaling

    grad_2d /= conf.c_SI
    grad_2d *= 200

    # TODO: smoothing

    return grad_2d

def slice(chi_i,grad_mesh,cosmo,conf):
    # slice the z(axis=2) range of the mesh
    # chi_i: where the light ray starts at this time step
    start_index = jnp.argmax(conf.mesh_chi>=chi_i) # will integrate the density fields that >= chi(a_start)
    return dynamic_slice_in_dim(grad_mesh, start_index, slice_size=conf.lens_mesh_shape[2], axis=2)