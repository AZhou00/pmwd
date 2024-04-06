from dataclasses import field
from functools import partial
from itertools import accumulate
from operator import itemgetter, mul
from typing import Optional, Any

from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.tree_util import pytree_dataclass
from pmwd.configuration import Configuration
from pmwd.cosmology import E2
from pmwd.util import is_float0_array
from pmwd.pm_util import enmesh

from pmwd.boltzmann import chi_a, r_a
from pmwd.particles import Particles

@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Rays:
    """ray state.

    Rays are indexable.

    Array-likes are converted to ``jax.Array`` of ``conf.pmid_dtype`` or
    ``conf.float_dtype`` at instantiation.

    For rays, the 3d coordinates are defined by the metric \diag(r^2, r^2, 1)(1-2\Phi) where
    r is the comoving angular diameter distance at the ray's comoving distance \chi, and \Phi is the
    Newtonian potential. The coordinaes and velocities are defined with respect to this metric, in particular
    they are not Cartesian. See \ref{} for details.

    Signs. We are at chi = 0. In real life, ray travels to smaller chi. During ray tracing, the tracing is towards
    larger chi.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    pmid : ArrayLike
        Particle(Ray) IDs by grid indices, of signed int dtype. They are the initial values of the rays'
        angular positions on the image plane. They also define the main line-of-sight (LOS) of each ray.
        `pos` is the 2d image plane position of the rays.
        `pos_3d` is the 3d comoving position of the rays.
    disp : ArrayLike
        Ray (image plane, angular) displacements (relative to the position specified by pmid) in [rad].
    eta: ArrayLike
        Rays' 2d angular momenta, r(\chi) v; v is the 2d comoving velocity perpendicular to each ray's main LOS.
        In this formalism, the normalized 3d momentum of a ray is given by (-v/r(\chi), -1+\order(v^2)), the signs
        are giving by the physical motion of the rays. Unit [H_0 L^2] 
    deta : ArrayLike 
        Rays' 2d deta (angular impulse) in unit of [H_0 L^2], 
        i.e., deta = \Delta eta = \partial eta / \partial \chi * \Delta \chi
    attr : pytree, optional
        Particle attributes (custom features).
    """

    conf: Configuration = field(repr=False)

    pmid: ArrayLike
    disp: ArrayLike  # image plane displacement, not 3D displacement
    eta: Optional[ArrayLike] = None # image plane angular momentum
    deta: Optional[ArrayLike] = None # image plane deta (angular impulse or Delta eta)
    A: Optional[ArrayLike] = None # the distortion matrix d theta_n / d theta_0
    B: Optional[ArrayLike] = None # d theta_eta / d theta_0
    dB: Optional[ArrayLike] = None
    attr: Any = None

    def __post_init__(self):
        if self._is_transforming():
            return

        conf = self.conf
        for name, value in self.named_children():
            dtype = conf.pmid_dtype if name == "pmid" else conf.float_dtype
            if name == "attr":
                value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            else:
                value = (
                    value
                    if value is None or is_float0_array(value)
                    else jnp.asarray(value, dtype=dtype)
                )

            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.pmid)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)

    @classmethod
    def gen_grid(cls, conf, eta=False, deta=False, A=False, B=False, dB=False):
        """
        Generate rays at z=0 on the pixelized grid. The velocity aligns with the positive positive time. 
        Only two components of the angular momentum are initialized, the third component is constrained by the mass-shell condition.
        i.e., am_z = 1 + \order(v^2)
        
        The variables and their /d_theta0 pairs are
        disp <-> A
        eta <-> B
        deta <-> dB
        
        Parameters
        ----------
        conf : Configuration
        eta : bool, optional
            Whether to initialize angular momenta to zeros.
        deta : bool, optional
            Whether to initialize twirls to zeros.
        A : bool, optional
            Whether to initialize distortion matrices (d theta / d theta_0) to the 2x2 identity matrix for each ray.
        B : bool, optional
            Whether to initialize distortion matrices (d eta_0 / d theta_0) to the 2x2 zero matrix for each ray.
        dB : bool, optional
            Whether to initialize distortion matrix gradients to zeros.
        """
        # print("fnc: Rays.gen_grid")
        pmid, disp = [], []
        for i, (sp, sm) in enumerate(zip(conf.ray_grid_shape, conf.ray_mesh_shape_default)):
            pmid_1d = jnp.linspace(0, sm, num=sp, endpoint=False)
            pmid_1d = jnp.rint(pmid_1d)
            pmid_1d = pmid_1d.astype(conf.pmid_dtype)
            pmid.append(pmid_1d)

            # exact int arithmetic, non-0 when mesh shape is not integer multiple of particle grid
            disp_1d = jnp.arange(sp) * sm - pmid_1d.astype(int) * sp
            disp_1d *= conf.ray_cell_size_default / sp
            disp_1d = disp_1d.astype(conf.float_dtype)
            disp.append(disp_1d)
        
        pmid = jnp.meshgrid(*pmid, indexing="ij")
        pmid = jnp.stack(pmid, axis=-1).reshape(-1, 2) # image plane is 2d, shape (ray_num, 2)
        pmid -= pmid.mean(axis=0) # define the image plane origin at the center of the grid

        disp = jnp.meshgrid(*disp, indexing="ij")
        disp = jnp.stack(disp, axis=-1).reshape(-1, 2) # image plane is 2d
        
        eta = jnp.zeros_like(disp, dtype=conf.float_dtype) if not eta else None
        deta = jnp.zeros_like(disp, dtype=conf.float_dtype) if not deta else None 
        A = jnp.eye(2, dtype=conf.float_dtype).reshape(1, 2, 2).repeat(len(pmid), axis=0) if not A else None
        B = jnp.zeros((len(pmid), 2, 2), dtype=conf.float_dtype) if not B else None
        dB = jnp.zeros_like(B, dtype=conf.float_dtype)
        # print(A.shape) # (ray_num, 2, 2)
        return cls(conf, pmid, disp, eta=eta, deta=deta, A=A, B=B, dB=dB)

    def pos_0(self, dtype=jnp.float32):
        """Ray positions on the 2d image plane at z=0.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Output float dtype.

        Returns
        -------
        pos : jax.Array
            Ray (image plane, angular) 2d positions in [rad].

        Notes
        -----
        There is no wrapping since we are working on angular coordinates.
        Wrapping only happens in 3d comoving coordinates.
        """
        pos = self.pmid.astype(dtype)
        pos *= self.conf.ray_cell_size_default
        return pos
    
    def pos_ip(self, dtype=jnp.float32): 
        #TODO: ask why jnp.64, ok probably becasue we are merging pmid and disp
        #TODO: maybe 32 for ray tracing is good enough. is this a major issue?
        """Ray positions on the 2d image plane angular 

        Parameters
        ----------
        dtype : DTypeLike, optional
            Output float dtype.

        Returns
        -------
        pos : jax.Array
            Ray (image plane, angular) 2d positions in [rad].

        Notes
        -----
        There is no wrapping since we are working on angular coordinates.
        Wrapping only happens in 3d comoving coordinates.
        """
        # origin is taken to be the center of the ray grid
        pos = self.pmid.astype(dtype)
        pos *= self.conf.ray_cell_size_default # note this is not the ray_spaing, but the mesh spacing
        pos += self.disp.astype(dtype) # disp in [rad]
        return pos
    
    def pos_3d(self, a, cosmo, conf, dtype=jnp.float32, wrap=True):
        """Ray 3d comoving positions, at the a=a slice.

        Parameters
        ----------
        a : float
            scale factor
        dtype : DTypeLike, optional
            Output float dtype.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        Returns
        -------
        pos : jax.Array
            Ray 3d comoving positions in [Mpc/h], shape (ray_num, 3)

        Notes
        -----
        The 3d comoving positions are computed from the 2d angular positions
        and the cosmology. The 3d positions are wrapped around the periodic
        boundaries if ``wrap`` is ``True``.
        """
        conf = self.conf
        
        pos = self.pos_ip(dtype) * r_a(a, cosmo, conf) # shape (ray_num, 2)
        pos += conf.ray_origin # observer origin at the center of the particle mesh at z=0

        pos_3d = jnp.pad(pos, ((0, 0), (0, 1)), constant_values=chi_a(a, cosmo, conf))

        if wrap:
            pos_3d %= jnp.array(conf.box_size, dtype=dtype)

        return pos_3d

    # def gen_traj(self, a0, a1, cosmo, conf, ):
    #     """Generate a particle grid using the Particle object such that the particle positions 
    #     interpolate the trajectory of the ray grid between two lens planes

    #     Parameters
    #     ----------
    #     a0 : float
    #         Scale factor of the first lens plane
    #     a1 : float
    #         Scale factor of the second lens plane
    #     conf : Configuration

    #     Returns
    #     -------
    #     traj : Particles
    #         Particle object that interpolates the ray grid between two lens planes
    #     """
    #     print("fnc: Rays.gen_traj")
    #     a_list = jnp.linspace(a0, a1, 5) # the integration will happen in 5 steps (will need to remove loop and make step size smaller)
    #     traj = []
    #     for a in a_list: # TODO: remove loop
    #         # chi = distance(a, cosmo, conf)
    #         pos_3d = self.pos_3d(a, cosmo)
    #         traj.append(pos_3d)
    #     print(traj[0].shape)
    #     traj = jnp.stack(traj, axis=-1) # shape (ray_num, 3, num_integration steps)
    #     traj = jnp.swapaxes(traj, 1, 2) # shape (ray_num, num_integration steps, 3)

    #     return traj

    # def gen_los_traj_mesh(self, a0, a1, cosmo, conf, ):
    #     a_list = jnp.linspace(a0, a1, 5) # the integration will happen in 5 steps (will need to remove loop and make step size smaller)
    #     traj = []
    #     for a in a_list: # TODO: remove loop
    #         # chi = distance(a, cosmo, conf)
    #         pos_3d = self.pos_3d(a, cosmo)
    #         traj.append(pos_3d)
    #     print(traj[0].shape)
    #     traj = jnp.stack(traj, axis=-1) # shape (ray_num, 3, num_integration steps)
    #     traj = jnp.swapaxes(traj, 1, 2) # shape (ray_num, num_integration steps, 3)

    #     return traj

    

    #     # traj_grid_shape = (self.ray_grid_shape[0], self.ray_grid_shape[1], 10)
    #     # ptcl_spacing = self.
    #     # conf_traj = Configuration(ptcl_spacing, traj_grid_shape, conf.ray_spacing, conf.ray_grid_shape, mesh_shape=conf.mesh_shape)
        
    #     # traj = Particles.gen_grid(conf, vel=True)
    #     # pos = traj.pos()

    #     # traj.replace(disp=disp)





