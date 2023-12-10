from functools import partial
import math
from typing import ClassVar, Optional, Tuple, Union

import jax
from jax import ensure_compile_time_eval
from jax.typing import DTypeLike
import jax.numpy as jnp
from jax.tree_util import tree_map
from mcfit import TophatVar

from pmwd.tree_util import pytree_dataclass


jax.config.update("jax_enable_x64", True)


jnp.set_printoptions(precision=3, edgeitems=2, linewidth=128)


@partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True)
class Configuration:
    """Configuration parameters, "immutable" as a frozen dataclass.

    Parameters
    ----------
    ptcl_spacing : float
        Lagrangian particle grid cell size in [L].
    ptcl_grid_shape : tuple of int
        Lagrangian particle grid shape, in ``len(ptcl_grid_shape)`` spatial dimensions.
    ray_grid_shape : tuple of int
        tuple of length 2
    mesh_shape : int, float, or tuple of int, optional
        Mesh shape. If an int or float, it is used as the 1D mesh to particle grid shape
        ratio, to determine the mesh shape from that of the particle grid. The mesh grid
        cannot be smaller than the particle grid (int or float values must not be
        smaller than 1) and the two grids must have the same aspect ratio.
    cosmo_dtype : DTypeLike, optional
        Float dtype for Cosmology and Configuration.
    pmid_dtype : DTypeLike, optional
        Signed integer dtype for particle or mesh grid indices.
    float_dtype : DTypeLike, optional
        Float dtype for other particle and mesh quantities.
    k_pivot_Mpc : float, optional
        Primordial scalar power spectrum pivot scale in 1/Mpc.
    T_cmb : float, optional
        CMB temperature in K.
    M : float, optional
        Mass unit defined in kg/h. Default is 1e10 M☉/h.
    L : float, optional
        Length unit defined in m/h. Default is Mpc/h.
    T : float, optional
        Time unit defined in s/h. Default is Hubble time 1/H_0 ~ 1e10 years/h ~ age of
        the Universe.
    transfer_fit : bool, optional
        Whether to use Eisenstein & Hu fit to transfer function. Default is True
        (subject to change when False is implemented).
    transfer_fit_nowiggle : bool, optional
        Whether to use non-oscillatory transfer function fit.
    transfer_lgk_min : float, optional
        Minimum transfer function wavenumber in [1/L] in log10.
    transfer_lgk_max : float, optional
        Maximum transfer function wavenumber in [1/L] in log10.
    transfer_lgk_maxstep : float, optional
        Maximum transfer function wavenumber step size in [1/L] in log10. It determines
        the number of wavenumbers ``transfer_k_num``, the actual step size
        ``transfer_lgk_step``, and the wavenumbers ``transfer_k``.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs.
    growth_inistep: float, None, or 2-tuple of float or None, optional
        The initial step size for solving the growth ODEs. If None, use estimation. If a
        tuple, use the two step sizes for forward and reverse integrations,
        respectively.
    lpt_order : int, optional
        LPT order, with 1 for Zel'dovich approximation, 2 for 2LPT, and 3 for 3LPT.
    a_start : float, optional
        LPT scale factor and N-body starting time.
    a_stop : float, optional
        N-body stopping time (scale factor).
    a_lpt_maxstep : float, optional
        Maximum LPT light cone scale factor step size. It determines the number of steps
        ``a_lpt_num``, the actual step size ``a_lpt_step``, and the steps ``a_lpt``.
    a_nbody_maxstep : float, optional
        Maximum N-body time integration scale factor step size. It determines the number
        of steps ``a_nbody_num``, the actual step size ``a_nbody_step``, and the steps
        ``a_nbody``.
    symp_splits : tuple of float 2-tuples, optional
        Symplectic splitting method composition, with each 2-tuples being drift and then
        kick coefficients. Its adjoint has the same splits in reverse nested orders,
        i.e., kick and then drift. Default is the Newton-Störmer-Verlet-leapfrog method.
    chunk_size : int, optional
        Chunk size to split particles in batches in scatter and gather to save memory.

    Raises
    ------
    ValueError
        Incorrect or inconsistent parameter values.

    """

    ptcl_spacing: float
    ptcl_grid_shape: Tuple[int, ...]  # tuple[int, ...] for python >= 3.9 (PEP 585)
    
    ray_spacing: float
    ray_grid_shape: Tuple[int, ...]

    mesh_shape: Union[float, Tuple[int, ...]] = 1
    ray_mesh_shape: Union[float, Tuple[int, ...]] = 1

    cosmo_dtype: DTypeLike = jnp.float64
    pmid_dtype: DTypeLike = jnp.int16
    float_dtype: DTypeLike = jnp.float32

    k_pivot_Mpc: float = 0.05

    T_cmb: float = 2.7255  # Fixsen 2009, arXiv:0911.1955

    # constants in SI units, as class variables
    M_sun_SI: ClassVar[float] = 1.98847e30  # solar mass in kg
    Mpc_SI: ClassVar[float] = 3.0856775815e22  # Mpc in m
    H_0_SI: ClassVar[float] = 1e5 / Mpc_SI  # Hubble constant in h/s
    c_SI: ClassVar[int] = 299792458  # speed of light in m/s
    G_SI: ClassVar[float] = 6.67430e-11  # Gravitational constant in m^3/kg/s^2

    # Units
    M: float = 1e10 * M_sun_SI
    L: float = Mpc_SI
    T: float = 1 / H_0_SI

    transfer_fit: bool = True
    transfer_fit_nowiggle: bool = False
    transfer_lgk_min: float = -4
    transfer_lgk_max: float = 3
    transfer_lgk_maxstep: float = 1/128

    growth_rtol: Optional[float] = None
    growth_atol: Optional[float] = None
    growth_inistep: Union[float, None,
                          Tuple[Optional[float], Optional[float]]] = (1, None)

    lpt_order: int = 2

    a_start: float = 1/64
    a_stop: float = 1
    a_lpt_maxstep: float = 1/128
    a_nbody_maxstep: float = 1/64

    symp_splits: Tuple[Tuple[float, float], ...] = ((0, 0.5), (1, 0.5))

    chunk_size: int = 2**24

    # other ray tracing parameters, limiting z(a) for ray tracing. i.e., furthest source location 
    z_rtlim: float = 2
    a_rtlim: float = 1 / (1 + z_rtlim)
    
    def __post_init__(self):
        if self._is_transforming():
            return

        # parse mesh_shape
        if isinstance(self.mesh_shape, (int, float)):
            mesh_shape = tuple(round(s * self.mesh_shape) for s in self.ptcl_grid_shape)
            object.__setattr__(self, 'mesh_shape', mesh_shape)
        if len(self.ptcl_grid_shape) != len(self.mesh_shape):
            raise ValueError('particle and mesh grid dimensions differ')
        if any(sm < sp for sp, sm in zip(self.ptcl_grid_shape, self.mesh_shape)):
            raise ValueError('mesh grid cannot be smaller than particle grid')
        if any(self.ptcl_grid_shape[0] * sm != self.mesh_shape[0] * sp
               for sp, sm in zip(self.ptcl_grid_shape[1:], self.mesh_shape[1:])):
            raise ValueError('particle and mesh grid aspect ratios differ')
        
        # parse ray_mesh_shape
        if isinstance(self.ray_mesh_shape, (int, float)):
            ray_mesh_shape = tuple(round(s * self.ray_mesh_shape) for s in self.ray_grid_shape)
            object.__setattr__(self, 'ray_mesh_shape', ray_mesh_shape)
        if len(self.ray_grid_shape) != len(self.ray_mesh_shape):
            raise ValueError('ray and mesh grid dimensions differ')
        if any(sm < sp for sp, sm in zip(self.ray_grid_shape, self.ray_mesh_shape)):
            raise ValueError('mesh grid cannot be smaller than particle grid')
        if any(self.ray_grid_shape[0] * sm != self.ray_mesh_shape[0] * sp
               for sp, sm in zip(self.ray_grid_shape[1:], self.ray_mesh_shape[1:])):
            raise ValueError('ray and mesh grid aspect ratios differ')

        object.__setattr__(self, 'cosmo_dtype', jnp.dtype(self.cosmo_dtype))
        object.__setattr__(self, 'pmid_dtype', jnp.dtype(self.pmid_dtype))
        object.__setattr__(self, 'float_dtype', jnp.dtype(self.float_dtype))
        if not jnp.issubdtype(self.cosmo_dtype, jnp.floating):
            raise ValueError('cosmo_dtype must be floating point numbers')
        if not jnp.issubdtype(self.pmid_dtype, jnp.signedinteger):
            raise ValueError('pmid_dtype must be signed integers')
        if not jnp.issubdtype(self.float_dtype, jnp.floating):
            raise ValueError('float_dtype must be floating point numbers')

        with jax.ensure_compile_time_eval():
            object.__setattr__(
                self,
                'var_tophat',
                TophatVar(self.transfer_k[1:], lowring=True, backend='jax'),
            )

        # ~ 1.5e-8 for float64, 3.5e-4 for float32
        growth_tol = math.sqrt(jnp.finfo(self.cosmo_dtype).eps)
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol', growth_tol)
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol', growth_tol)

        if any(len(s) != 2 for s in self.symp_splits):
            raise ValueError(f'symp_splits={self.symp_splits} not supported')
        symp_splits_sum = tuple(sum(s) for s in zip(*self.symp_splits))
        if symp_splits_sum != (1, 1):
            raise ValueError(f'sum of symplectic splits = {symp_splits_sum} != (1, 1)')

        dtype = self.cosmo_dtype
        for name, value in self.named_children():
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, name, value)

    @property
    def dim(self):
        """Spatial dimension."""
        return len(self.ptcl_grid_shape)

    @property
    def ptcl_cell_vol(self):
        """Lagrangian particle grid cell volume in [L^dim]."""
        return self.ptcl_spacing ** self.dim

    @property
    def ptcl_num(self):
        """Number of particles."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ptcl_grid_shape).prod().item()

    @property
    def ray_cell_area(self):
        """Pixel area in [rad^2]."""
        return self.ray_spacing ** 2

    @property
    def ray_num(self):
        """Number of rays."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ray_grid_shape).prod().item()
        
    @property
    def box_size(self):
        """Simulation box size tuple in [L]."""
        return tuple(self.ptcl_spacing * s for s in self.ptcl_grid_shape)

    @property
    def box_vol(self):
        """Simulation box volume in [L^dim]."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.box_size).prod().item()

    @property
    def cell_size(self):
        """Mesh cell size in [L]."""
        return self.ptcl_spacing * self.ptcl_grid_shape[0] / self.mesh_shape[0]

    @property
    def cell_vol(self):
        """Mesh cell volume in [L^dim]."""
        return self.cell_size ** self.dim

    @property
    def mesh_size(self):
        """Number of mesh grid points."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.mesh_shape).prod().item()

    @property
    def ray_cell_size(self):
        """image plane mesh cell size in [rad]."""
        return self.ray_spacing * self.ray_grid_shape[0] / self.ray_mesh_shape[0]
    
    @property
    def ray_cell_area(self):
        """image plane mesh cell area in [rad^2]."""
        return self.ray_cell_size ** 2
    
    @property
    def ray_mesh_size(self):
        """Number of image plane mesh grid points."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ray_mesh_shape).prod().item()
    
    @property #TODO: is there a difference between computing fov using rays/mesh?
    def ray_mesh_fov(self):
        """Field of view in [rad]."""
        return self.ray_cell_size * self.ray_mesh_shape[0], self.ray_cell_size * self.ray_mesh_shape[1]
    
    @property
    def V(self):
        """Velocity unit as [L/T]. Default is 100 km/s."""
        return self.L / self.T

    @property
    def H_0(self):
        """Hubble constant H_0 in [1/T]."""
        return self.H_0_SI * self.T

    @property
    def c(self):
        """Speed of light in [L/T]."""
        return self.c_SI / self.V

    @property
    def G(self):
        """Gravitational constant in [L^3 / M / T^2]."""
        return self.G_SI * self.M / (self.L * self.V**2)

    @property
    def rho_crit(self):
        """Critical density in [M / L^3]."""
        return 3 * self.H_0**2 / (8 * jnp.pi * self.G)

    @property
    def transfer_k_num(self):
        """Number of transfer function wavenumbers, including a leading 0."""
        return 1 + math.ceil((self.transfer_lgk_max - self.transfer_lgk_min)
                             / self.transfer_lgk_maxstep) + 1

    @property
    def transfer_lgk_step(self):
        """Transfer function wavenumber step size in [1/L] in log10."""
        return ((self.transfer_lgk_max - self.transfer_lgk_min)
                / (self.transfer_k_num - 2))

    @property
    def transfer_k(self):
        """Transfer function wavenumbers in [1/L], of ``cosmo_dtype``."""
        k = jnp.logspace(self.transfer_lgk_min, self.transfer_lgk_max,
                         num=self.transfer_k_num - 1, dtype=self.cosmo_dtype)
        return jnp.concatenate((jnp.array([0]), k))

    @property
    def a_lpt_num(self):
        """Number of LPT light cone scale factor steps, excluding ``a_start``."""
        return math.ceil(self.a_start / self.a_lpt_maxstep)

    @property
    def a_lpt_step(self):
        """LPT light cone scale factor step size."""
        return self.a_start / self.a_lpt_num

    @property
    def a_nbody_num(self):
        """Number of N-body time integration scale factor steps, excluding ``a_start``."""
        return math.ceil((self.a_stop - self.a_start) / self.a_nbody_maxstep)

    @property
    def a_nbody_step(self):
        """N-body time integration scale factor step size."""
        return (self.a_stop - self.a_start) / self.a_nbody_num

    @property
    def a_lpt(self):
        """LPT light cone scale factor steps, including ``a_start``, of ``cosmo_dtype``."""
        return jnp.linspace(0, self.a_start, num=self.a_lpt_num+1,
                            dtype=self.cosmo_dtype)

    @property
    def a_nbody(self):
        """N-body time integration scale factor steps, including ``a_start``, of ``cosmo_dtype``."""
        return jnp.linspace(self.a_start, self.a_stop, num=1+self.a_nbody_num,
                            dtype=self.cosmo_dtype)

    @property
    def a_nbody_ray(self):
        """N-body time integration scale factor steps for backward ray tracing"""
        index = jnp.argmax(self.a_nbody[::-1]<self.a_rtlim)
        return self.a_nbody[::-1][:index+1]
    
    @property
    def growth_a(self):
        """Growth function scale factors, for both LPT and N-body, of ``cosmo_dtype``."""
        return jnp.concatenate((self.a_lpt, self.a_nbody[1:]))

    @property
    def varlin_R(self):
        """Linear matter overdensity variance in a top-hat window of radius R in [L], of ``cosmo_dtype``."""
        return self.var_tophat.y

    @property
    def ray_origin(self):
        """The comoving location of the origin of the observer relative 
        to the particle mesh. This is used to shift the observer to the center of the x-y-(z=0) plane
        """
        return jnp.array([self.ptcl_grid_shape[0]*self.ptcl_spacing/2, 
                          self.ptcl_grid_shape[1]*self.ptcl_spacing/2])

    @property
    def mesh_chi(self):
        # comoving coordinates of the particle mesh in z direction 
        # observer at chi = 0, chi increases away from the observer
        return self.cell_size * jnp.arange(self.mesh_shape[2])
    
    @property
    def lens_mesh_shape(self):
        """Shape of the lens plane mesh grid.
        Given z_rtlim, the maximum thickness [number of mesh point] of the lens plane can be computed.
        The lens mesh is a slice of the particle mesh in z direction that covers the interval over which we will
        integrate the lensing potential. 
        """

        # chi = distance(self.a_nbody, cosmo, conf)
        # hard code for now:
        assert self.a_nbody.size == 64
        chi =  jnp.array([12185.4  , 11380.87 , 10761.901, 10239.586,  9779.245,  9363.047,  8980.4  ,  8624.407,  8290.289,  7974.573,  7674.657,
            7388.528,  7114.592,  6851.573,  6598.427,  6354.291,  6118.441,  5890.271,  5669.258,  5454.959,  5246.988,  5045.009,
            4848.729,  4657.887,  4472.253,  4291.62 ,  4115.804,  3944.635,  3777.959,  3615.635,  3457.532,  3303.526,  3153.503,
            3007.352,  2864.971,  2726.261,  2591.124,  2459.469,  2331.206,  2206.247,  2084.508,  1965.905,  1850.356,  1737.781,
            1628.101,  1521.24 ,  1417.122,  1315.671,  1216.816,  1120.484,  1026.605,   935.11 ,   845.931,   759.003,   674.261,
         591.641,   511.082,   432.524,   355.907,   281.175,   208.272,   137.143,    67.736,     0.   ])
        
        # the max slice size is defined by the comoving distance covered by one time step update
        # that includes z_rtlim
        chi_rtlim = jnp.interp(self.a_rtlim, self.a_nbody, chi) # comoving distance of the furthest source
        id_chi_upperbound = jnp.where(chi >= chi_rtlim)[0][-1] # chi is decreasingly sorted
        id_chi_lowerbound = jnp.where(chi <= chi_rtlim)[0][0] # chi is decreasingly sorted
        delta_chi = chi[id_chi_upperbound]-chi[id_chi_lowerbound] # Mpc/h
        len_z = int(delta_chi / self.cell_size * 2 / 3 ) # 1/2 should be good enough, but let's be safe for now
        return (self.mesh_shape[0], self.mesh_shape[1], len_z)
    
    @property
    def lens_mesh_size(self):
        """Number of mesh grid points in the lens plane."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.lens_mesh_shape).prod().item()
        
    
    # @property
    # def lens_slice_size(self):
    #     """Given z_rtlim, the maximum thickness [number of mesh point] of the lens plane.
    #     in fact we only need 1/2 of what is here, but just to be safe for now...
    #     """
    #     # chi = distance(self.a_nbody, cosmo, conf)
    #     # hard code for now:
    #     assert self.a_nbody.size == 64
    #     chi =  jnp.array([12185.4  , 11380.87 , 10761.901, 10239.586,  9779.245,  9363.047,  8980.4  ,  8624.407,  8290.289,  7974.573,  7674.657,
    #         7388.528,  7114.592,  6851.573,  6598.427,  6354.291,  6118.441,  5890.271,  5669.258,  5454.959,  5246.988,  5045.009,
    #         4848.729,  4657.887,  4472.253,  4291.62 ,  4115.804,  3944.635,  3777.959,  3615.635,  3457.532,  3303.526,  3153.503,
    #         3007.352,  2864.971,  2726.261,  2591.124,  2459.469,  2331.206,  2206.247,  2084.508,  1965.905,  1850.356,  1737.781,
    #         1628.101,  1521.24 ,  1417.122,  1315.671,  1216.816,  1120.484,  1026.605,   935.11 ,   845.931,   759.003,   674.261,
    #      591.641,   511.082,   432.524,   355.907,   281.175,   208.272,   137.143,    67.736,     0.   ])
        
    #     # the max slice size is defined by the comoving distance covered by one time step update
    #     # that includes z_rtlim
    #     chi_rtlim = jnp.interp(self.a_rtlim, self.a_nbody, chi) # comoving distance of the furthest source
    #     id_chi_upperbound = jnp.where(chi >= chi_rtlim)[0][-1] # chi is decreasingly sorted
    #     id_chi_lowerbound = jnp.where(chi <= chi_rtlim)[0][0] # chi is decreasingly sorted
    #     delta_chi = chi[id_chi_upperbound]-chi[id_chi_lowerbound] # Mpc/h
    #     return int(delta_chi/self.cell_size)