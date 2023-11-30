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

from pmwd.boltzmann import distance


@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Rays:
    """Particle state.

    Particles are indexable.

    Array-likes are converted to ``jax.Array`` of ``conf.pmid_dtype`` or
    ``conf.float_dtype`` at instantiation.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    pmid : ArrayLike
        Particle IDs by mesh indices, of signed int dtype. They are the initial values of the rays' Lagrangian 
        angular positions. The pmid's are mapped to angular coordinates. They also define the main 
        line-of-sight (LOS) of each ray. It can save memory compared to the raveled particle IDs, e.g., 
        6 bytes for 3 times int16 versus 8 bytes for uint64. `pos` is the 2d image plane position of the
        rays. `pos_3d` is the 3d comoving position of the rays.
    disp : ArrayLike
        # FIXME after adding the CUDA scatter and gather ops
        Ray (image plane, angular) displacements from pmid in [rad]. Call
        ``pos`` for the ray (image plane, angular) positions.
    vel : ArrayLike, optional
        Ray canonical (image plane, angular) velocities in [???]. 
    acc : ArrayLike, optional
        Ray canonical (image plane, angular) accelerations in [???].
    attr : pytree, optional
        Particle attributes (custom features).
    """

    conf: Configuration = field(repr=False)

    pmid: ArrayLike
    disp: ArrayLike # image coordinate displacement, not 3D displacement
    vel: Optional[ArrayLike] = None
    acc: Optional[ArrayLike] = None
    pmid_center: Optional[ArrayLike] = None
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
    def gen_grid(cls, conf, acc=False):
        """Generate rays on the a=1 observer plane on pixel grid.

        Parameters
        ----------
        conf : Configuration
        vel : bool, optional
            Whether to initialize velocities to zeros.
        acc : bool, optional
            Whether to initialize accelerations to zeros.

        """
        print('fnc: Rays.gen_grid')

        z_init = 0 # conf.cell_size * (conf.ptcl_grid_shape[-1] - 1)
        print('initiating rays on the plane Cartesian z =', z_init)

        pmid, disp = [], []
        for i, (sp, sm) in enumerate(zip(conf.ray_grid_shape, conf.ray_grid_shape)):
            # shape of particle of shape of ray mesh
            pmid_1d = jnp.linspace(0, sm, num=sp, endpoint=False)
            pmid_1d = jnp.rint(pmid_1d)
            pmid_1d = pmid_1d.astype(conf.pmid_dtype)
            pmid.append(pmid_1d)

            # exact int arithmetic
            # disp_1d = jnp.arange(sp) * sm - pmid_1d.astype(int) * sp
            # disp_1d *= conf.cell_size / sp
            # just 0
            disp_1d = jnp.zeros(sp)
            disp_1d = disp_1d.astype(conf.float_dtype)
            disp.append(disp_1d)
        # print('pmid', pmid)
        # print('disp', disp)

        pmid = jnp.meshgrid(*pmid, indexing="ij")
        pmid = jnp.stack(pmid, axis=-1).reshape(-1, conf.dim-1)
        pmid_center = pmid.mean(axis=0)

        disp = jnp.meshgrid(*disp, indexing="ij")
        disp = jnp.stack(disp, axis=-1).reshape(-1, conf.dim-1)

        # print(pmid.shape, disp.shape)
        
        # unit of ???
        # velocity: the direction is by default point towards the observer (opposite of the tracing direction), hence omitted.
        vel = jnp.zeros_like(disp) 
        acc = jnp.zeros_like(disp) if acc else None

        return cls(conf, pmid, disp, vel=vel, acc=acc, pmid_center=pmid_center)

    def pos(self, dtype=jnp.float64):
        """Ray (image plane, angular) positions.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Output float dtype.

        Returns
        -------
        pos : jax.Array
            Ray (image plane, angular) positions in [rad].

        Notes
        -----
        There is no wrapping since we are working on angular coordinates.
        Wrapping only happens in 3d comoving coordinates.
        """
        conf = self.conf

        pos = self.pmid.astype(dtype) - self.pmid_center.astype(dtype)
        pos *= conf.ray_spacing
        pos += self.disp.astype(dtype)
        
        return pos
    
    def pos_3d(self, a, cosmo, dtype=jnp.float64, wrap=True):
        """Ray 3d comoving positions, at the a=a slice.

        Parameters
        ----------
        a : float
            Scale factor.
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

        chi = distance(a, cosmo, conf)

        # x,y coords
        pos = self.pos(dtype) * chi
        pos += conf.obsv_origin_cmv

        # z coord
        pos_3d = jnp.pad(pos, ((0, 0), (0, 1)), constant_values=chi)

        if wrap:
            pos_3d %= jnp.array(conf.box_size, dtype=dtype)

        return pos_3d
