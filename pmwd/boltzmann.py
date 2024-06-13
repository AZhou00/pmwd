from jax import jit, custom_vjp, ensure_compile_time_eval, vmap
import jax.numpy as jnp

from pmwd.cosmology import H_deriv, Omega_m_a
from pmwd.ode_util import odeint
from jax.scipy.integrate import trapezoid


@jit
def chi_integ(a_start, cosmo, conf):
    """
    Calculate the comoving distance from the observer at a=1 to a given scale factor in a flat LambdaCDM universe.

    Parameters:
    - a: Scale factor of the universe.
    - H0: Hubble constant at the current time in km/s/Mpc.
    - Omega_m: Matter density parameter.
    - Omega_r: Radiation density parameter.
    - Omega_de: Dark energy density parameter.
    - c: Speed of light in km/s (default is 299792.458 km/s).

    Returns:
    - Comoving distance in Mpc.
    """
    Omega_k = cosmo.Omega_k
    Omega_m = cosmo.Omega_m
    Omega_r = 0  # TODO: cosmo.Omega_r
    Omega_de = cosmo.Omega_de

    a = jnp.linspace(a_start, 1, 1000)
    H_a = jnp.sqrt(Omega_m * a**-3 + Omega_r * a**-4 + Omega_k * a**-2 + Omega_de)
    integrand = conf.c / (a**2 * H_a)
    integral = trapezoid(integrand, x=a)

    return integral


def chi_tab(cosmo, conf):
    """Compute the comoving distances chi(a)"""
    distance_f = jit(vmap(chi_integ, in_axes=(0, None, None)))
    distance = distance_f(conf.distance_a, cosmo, conf)
    return cosmo.replace(chi=distance)


def r_tab(cosmo, conf):
    """Compute the comoving distance r(a)"""  # TODO this is only flat universe
    distance_f = jit(vmap(chi_integ, in_axes=(0, None, None)))
    distance = distance_f(conf.distance_a, cosmo, conf)
    return cosmo.replace(r=distance)


def AD_tab(cosmo, conf):
    """Compute the angular diameter distance r(a)"""  # TODO this is only flat universe
    distance_f = jit(vmap(chi_integ, in_axes=(0, None, None)))
    distance = distance_f(conf.distance_a, cosmo, conf)
    distance *= conf.distance_a
    return cosmo.replace(AD=distance)


def chi_a(a, cosmo, conf):
    """Compute the comoving distances chi(a)"""
    if cosmo.chi is None:
        raise ValueError("Distance table is empty. Call chi_tab or boltzmann first.")

    a = jnp.asarray(a)
    float_dtype = jnp.promote_types(a.dtype, float)

    chi = jnp.interp(a, conf.distance_a, cosmo.chi)

    return chi.astype(float_dtype)


def r_a(a, cosmo, conf):
    """Compute the radial angular diameter distance r(a)"""
    if cosmo.r is None:
        raise ValueError("Distance table is empty. Call r_tab or boltzmann first.")

    a = jnp.asarray(a)
    float_dtype = jnp.promote_types(a.dtype, float)

    r = jnp.interp(a, conf.distance_a, cosmo.r)

    return r.astype(float_dtype)


def AD_a(a, cosmo, conf):
    """Compute the angular diameter distance AD(a)"""
    if cosmo.AD is None:
        raise ValueError("Distance table is empty. Call AD_tab or boltzmann first.")

    a = jnp.asarray(a)
    float_dtype = jnp.promote_types(a.dtype, float)

    AD = jnp.interp(a, conf.distance_a, cosmo.AD)

    return AD.astype(float_dtype)


def a_chi(chi, cosmo, conf):
    """Compute the scale factor a(chi)"""
    if cosmo.chi is None:
        raise ValueError("Distance table is empty. Call chi_tab or boltzmann first.")

    chi = jnp.asarray(chi)
    float_dtype = jnp.promote_types(chi.dtype, float)

    # need to sort xp in ascending order
    a = jnp.interp(chi, cosmo.chi[::-1], conf.distance_a[::-1])

    return a.astype(float_dtype)


def r_chi(chi, cosmo, conf):
    """Compute the radial angular diameter distance r(chi)"""
    ## TODO, stable?
    a = a_chi(chi, cosmo, conf)
    r = r_a(a, cosmo, conf)
    return r


def growth_chi(chi, cosmo, conf, order=1, deriv=0):
    """Compute the growth function D(chi)"""
    a = conf.distance_a
    D_interp = growth(a, cosmo, conf, order, deriv)
    chi_interp = chi_a(a, cosmo, conf)
    D = jnp.interp(chi, chi_interp[::-1], D_interp[::-1])
    return D


@jit
def transfer_integ(cosmo, conf):
    """Compute and tabulate the transfer function at ``conf.transfer_k``.

    Parameters
    ----------
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a transfer table, that has the shape
        ``(conf.transfer_k_num,)`` and ``conf.cosmo_dtype``.

    """
    if conf.transfer_fit:
        transfer = transfer_fit(conf.transfer_k, cosmo, conf)
        return cosmo.replace(transfer=transfer)
    else:
        raise NotImplementedError('TODO')


# TODO Wayne's website: neutrino no wiggle case
def transfer_fit(k, cosmo, conf):
    """Eisenstein & Hu fit of matter transfer function at given wavenumbers.

    Parameters
    ----------
    k : ArrayLike
        Wavenumbers in [1/L].
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    T : jax.Array of (k * 1.).dtype
        Matter transfer function.

    .. _Transfer Function:
        http://background.uchicago.edu/~whu/transfer/transferpage.html

    """
    k = jnp.asarray(k)
    float_dtype = jnp.promote_types(k.dtype, float)

    k = k * cosmo.h / conf.L * conf.Mpc_SI  # unit conversion to [1/Mpc]

    T2_cmb_norm = (conf.T_cmb / 2.7)**2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2
    f_b = cosmo.Omega_b / cosmo.Omega_m
    f_c = cosmo.Omega_c / cosmo.Omega_m  # TODO neutrinos?

    z_eq = 2.50e4 * w_m / T2_cmb_norm**2
    k_eq = 7.46e-2 * w_m / T2_cmb_norm

    b1 = 0.313 * w_m**-0.419 * (1 + 0.607 * w_m**0.674)
    b2 = 0.238 * w_m**0.223
    z_d = 1291 * w_m**0.251 / (1 + 0.659 * w_m**0.828) * (1 + b1 * w_b**b2)

    R_d = 31.5 * w_b / T2_cmb_norm**2 * (1e3 / z_d)
    R_eq = 31.5 * w_b / T2_cmb_norm**2 * (1e3 / z_eq)
    s = (
        2 / (3 * k_eq) * jnp.sqrt(6 / R_eq)
        * jnp.log((jnp.sqrt(1 + R_d) + jnp.sqrt(R_eq + R_d)) / (1 + jnp.sqrt(R_eq)))
    )
    k_silk = 1.6 * w_b**0.52 * w_m**0.73 * (1 + (10.4 * w_m)**-0.95)

    if conf.transfer_fit_nowiggle:
        alpha_gamma = (1 - 0.328 * jnp.log(431 * w_m) * f_b
                       + 0.38 * jnp.log(22.3 * w_m) * f_b**2)
        gamma_eff_ratio = alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k * s)**4)

        q_eff = k / (13.41 * k_eq * gamma_eff_ratio)

        L0 = jnp.log(2 * jnp.e + 1.8 * q_eff)
        C0 = 14.2 + 731 / (1 + 62.5 * q_eff)
        T0 = L0 / (L0 + C0 * q_eff**2)

        return T0

    a1 = (46.9 * w_m)**0.670 * (1 + (32.1 * w_m)**-0.532)
    a2 = (12.0 * w_m)**0.424 * (1 + (45.0 * w_m)**-0.582)
    alpha_c = a1**-f_b * a2**-f_b**3
    b1 = 0.944 / (1 + (458 * w_m)**-0.708)
    b2 = (0.395 * w_m)**-0.0266
    beta_c = 1 / (1 + b1 * (f_c**b2 - 1))

    def T0_tilde(k, alpha_c, beta_c):
        q = k / (13.41 * k_eq)
        L = jnp.log(jnp.e + 1.8 * beta_c * q)
        C = 14.2 / alpha_c + 386 / (1 + 69.9 * q**1.08)
        T0 = L / (L + C * q**2)
        return T0

    f = 1 / (1 + (k * s / 5.4)**4)
    T_c = f * T0_tilde(k, 1, beta_c) + (1 - f) * T0_tilde(k, alpha_c, beta_c)

    y = (1 + z_eq) / (1 + z_d)
    x = jnp.sqrt(1 + y)
    G = y * (-6 * x + (2 + 3 * y) * jnp.log((x + 1) / (x - 1)))
    alpha_b = 2.07 * k_eq * s * (1 + R_d)**-0.75 * G

    beta_node = 8.41 * w_m**0.435
    beta_b = 0.5 + f_b + (3 - 2 * f_b) * jnp.sqrt(1 + (17.2 * w_m)**2)

    T_b = (
        T0_tilde(k, 1, 1) / (1 + (k * s / 5.2)**2)
        + alpha_b * (k * s)**3 / (beta_b**3 + (k * s)**3) * jnp.exp(-(k / k_silk)**1.4)
    ) * jnp.sinc((k * s)**2 / (jnp.pi * jnp.cbrt(beta_node**3 + (k * s)**3)))

    T = f_c * T_c + f_b * T_b

    return T.astype(float_dtype)


def transfer(k, cosmo, conf):
    """Evaluate interpolation or Eisenstein & Hu fit of matter transfer function at
    given wavenumbers.

    Parameters
    ----------
    k : ArrayLike
        Wavenumbers in [1/L].
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    T : jax.Array of (k * 1.).dtype
        Matter transfer function.

    Raises
    ------
    ValueError
        If ``cosmo.transfer`` table is empty.

    """
    if cosmo.transfer is None:
        raise ValueError('Transfer table is empty. '
                         'Call transfer_integ or boltzmann first.')

    k = jnp.asarray(k)
    float_dtype = jnp.promote_types(k.dtype, float)

    if conf.transfer_fit:
        T = jnp.interp(k, conf.transfer_k, cosmo.transfer)
    else:
        raise NotImplementedError('TODO')

    return T.astype(float_dtype)


@jit
def growth_integ(cosmo, conf):
    """Integrate and tabulate (LPT) growth functions and derivatives at
    ``conf.growth_a``.

    Parameters
    ----------
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a growth table, that has the shape ``(num_lpt_order,
        num_derivatives, len(conf.growth_a))`` and ``conf.cosmo_dtype``.

    Notes
    -----

    TODO: ODE math

    """
    with ensure_compile_time_eval():
        eps = jnp.finfo(conf.cosmo_dtype).eps
        a_ic = 0.5 * jnp.cbrt(eps).item()  # ~ 3e-6 for float64, 2e-3 for float32
        if a_ic >= conf.a_lpt_step:
            a_ic = 0.1 * conf.a_lpt_step

    a = conf.growth_a
    lna = jnp.log(a.at[0].set(a_ic))

    num_order, num_deriv, num_a = 2, 3, len(a)

    # TODO necessary to add lpt_order support?
    # G and lna can either be at a single time, or have leading time axes
    def ode(G, lna, cosmo):
        a = jnp.exp(lna)
        dlnH_dlna = H_deriv(a, cosmo)
        Omega_fac = 1.5 * Omega_m_a(a, cosmo)
        G1, G1p, G2, G2p = jnp.split(G, num_order * (num_deriv-1), axis=-1)
        G1pp = -(3 + dlnH_dlna - Omega_fac) * G1 - (4 + dlnH_dlna) * G1p
        G2pp = Omega_fac * G1**2 - (8 + 2*dlnH_dlna - Omega_fac) * G2 - (6 + dlnH_dlna) * G2p
        return jnp.concatenate((G1p, G1pp, G2p, G2pp), axis=-1)

    G_ic = jnp.array((1, 0, 3/7, 0), dtype=conf.cosmo_dtype)

    G = odeint(ode, G_ic, lna, cosmo,
               rtol=conf.growth_rtol, atol=conf.growth_atol, dt0=conf.growth_inistep)

    G_deriv = ode(G, lna[:, jnp.newaxis], cosmo)

    G = G.reshape(num_a, num_order, num_deriv-1)
    G_deriv = G_deriv.reshape(num_a, num_order, num_deriv-1)
    G = jnp.concatenate((G, G_deriv[..., -1:]), axis=2)
    G = jnp.moveaxis(G, 0, 2)

    # D_m /a^m = G
    # D_m'/a^m = m G + G'
    # D_m"/a^m = m^2 G + 2m G' + G"
    m = jnp.array((1, 2), dtype=conf.cosmo_dtype)[:, jnp.newaxis]
    growth = jnp.stack((
        G[:, 0],
        m * G[:, 0] + G[:, 1],
        m**2 * G[:, 0] + 2 * m * G[:, 1] + G[:, 2],
    ), axis=1)

    return cosmo.replace(growth=growth)


# TODO 3rd order has two factors, so `order` probably need to support str
def growth(a, cosmo, conf, order=1, deriv=0):
    """Evaluate interpolation of (LPT) growth function or derivative, the n-th
    derivatives of the m-th order growth function :math:`\mathrm{d}^n D_m /
    \mathrm{d}\ln^n a`, at given scale factors. Growth functions are normalized at the
    matter dominated era instead of today.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology
    conf : Configuration
    order : int in {1, 2}, optional
        Order of growth function.
    deriv : int in {0, 1, 2}, optional
        Order of growth function derivatives.

    Returns
    -------
    D : jax.Array of (a * 1.).dtype
        Growth functions or derivatives.

    Raises
    ------
    ValueError
        If ``cosmo.growth`` table is empty.

    """
    if cosmo.growth is None:
        raise ValueError('Growth table is empty. Call growth_integ or boltzmann first.')

    a = jnp.asarray(a)
    float_dtype = jnp.promote_types(a.dtype, float)

    D = a**order * jnp.interp(a, conf.growth_a, cosmo.growth[order-1][deriv])

    return D.astype(float_dtype)


def varlin_integ(cosmo, conf):
    """Compute and tabulate the linear matter overdensity variance at ``conf.varlin_R``.

    Parameters
    ----------
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a linear variance table, that has the shape
        ``(len(conf.varlin_R),)`` and ``conf.cosmo_dtype``.

    """
    Plin = linear_power(conf.var_tophat.x, None, cosmo, conf)

    _, varlin = conf.var_tophat(Plin, extrap=True)

    return cosmo.replace(varlin=varlin)


def varlin(R, a, cosmo, conf):
    """Evaluate interpolation of linear matter overdensity variance at given scales and
    scale factors.

    Parameters
    ----------
    R : ArrayLike
        Scales in [L].
    a : ArrayLike or None
        Scale factors. If None, output is not scaled by growth.
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    sigma2 : jax.Array of (k * a * 1.).dtype
        Linear matter overdensity variance.

    Raises
    ------
    ValueError
        If ``cosmo.varlin`` table is empty.

    """
    if cosmo.varlin is None:
        raise ValueError('Linear matter overdensity variance table is empty. '
                         'Call varlin_integ or boltzmann first.')

    R = jnp.asarray(R)
    float_dtype = jnp.promote_types(R.dtype, float)

    sigma2 = jnp.interp(R, conf.varlin_R, cosmo.varlin)

    if a is not None:
        a = jnp.asarray(a)
        float_dtype = jnp.promote_types(float_dtype, a.dtype)

        D = growth(a, cosmo, conf)

        sigma2 *= D**2

    return sigma2.astype(float_dtype)


def boltzmann(cosmo, conf, transfer=True, growth=True, varlin=True, distance=True, mesh=True):
    """Solve Einstein-Boltzmann equations and precompute transfer and growth functions,
    etc.

    Parameters
    ----------
    cosmo : Cosmology
    conf : Configuration
    transfer : bool, optional
        Whether to compute the transfer function, or to set it to None.
    growth : bool, optional
        Whether to compute the growth functions, or to set it to None.
    varlin : bool, optional
        Whether to compute the linear matter overdensity variance, or to set it to None.
    distance : bool, optional
        Whether to compute the lens weight function, or to set it to None.

    Returns
    -------
    cosmo : Cosmology
        A new instance containing transfer and growth tables, etc.

    """
    if transfer:
        cosmo = transfer_integ(cosmo, conf)
    else:
        cosmo = cosmo.replace(transfer=None)

    if growth:
        cosmo = growth_integ(cosmo, conf)
    else:
        cosmo = cosmo.replace(growth=None)

    if varlin:
        cosmo = varlin_integ(cosmo, conf)
    else:
        cosmo = cosmo.replace(varlin=None)

    if distance:
        cosmo = chi_tab(cosmo, conf)
        cosmo = r_tab(cosmo, conf)

    if mesh:
        cosmo = precompute_mesh(conf, cosmo) 
        # probably belongs to conf but this computation needs distance function to be ran first
        # will move this later
    return cosmo

@custom_vjp
def _safe_power(x1, x2):
    """Safe power function for x1==0 and 0<x2<1. x2 must be a scalar."""
    return x1 ** x2

def _safe_power_fwd(x1, x2):
    y = _safe_power(x1, x2)
    return y, (x1, x2, y)

def _safe_power_bwd(res, y_cot):
    x1, x2, y = res

    x1_cot = jnp.where(x1 != 0, x2 * y / x1 * y_cot, 0)

    lnx1 = jnp.where(x1 != 0, jnp.log(x1), 0)
    x2_cot = (lnx1 * y * y_cot).sum()

    return x1_cot, x2_cot

_safe_power.defvjp(_safe_power_fwd, _safe_power_bwd)


def linear_power(k, a, cosmo, conf):
    r"""Linear matter power spectrum at given wavenumbers and scale factors.

    Parameters
    ----------
    k : ArrayLike
        Wavenumbers in [1/L].
    a : ArrayLike or None
        Scale factors. If None, output is not scaled by growth.
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    Plin : jax.Array of (k * a * 1.).dtype
        Linear matter power spectrum in [L^3].

    Raises
    ------
    ValueError
        If not in 3D.

    Notes
    -----

    .. math::

        \frac{k^3}{2\pi^2} P_\mathrm{lin}(k, a)
        = \frac{4}{25} A_\mathrm{s}
            \Bigl( \frac{k}{k_\mathrm{pivot}} \Bigr)^{n_\mathrm{s} - 1}
            T^2(k)
            \Bigl( \frac{c k}{H_0} \Bigr)^4
            \Bigl( \frac{D(a)}{\Omega_\mathrm{m}} \Bigr)^2

    """
    if conf.dim != 3:
        raise ValueError(f'dim={conf.dim} not supported')

    k = jnp.asarray(k)
    float_dtype = jnp.promote_types(k.dtype, float)

    T = transfer(k, cosmo, conf)

    Plin = (
        0.32 * cosmo.A_s * cosmo.k_pivot * _safe_power(k / cosmo.k_pivot, cosmo.n_s)
        * (jnp.pi * (conf.c / conf.H_0)**2 / cosmo.Omega_m * T)**2
    )

    if a is not None:
        a = jnp.asarray(a)
        float_dtype = jnp.promote_types(float_dtype, a.dtype)

        D = growth(a, cosmo, conf)

        Plin *= D**2

    return Plin.astype(float_dtype)

def precompute_mesh(conf, cosmo):
    from pmwd.boltzmann import r_a
    from pmwd.ray_mesh import compute_ray_mesh
    
    a_nbody = conf.a_nbody_rt
    ray_cell_size_list = []
    ray_mesh_shape_list = []
    for a_prev, a_next in zip(a_nbody[:-1], a_nbody[1:]):
        def integrate_ray(a_prev, a_next, cosmo, conf):
            D = K = 0
            a_disp = a_vel = a_prev
            
            # KDK
            for d, k in conf.symp_splits:
                if d != 0:
                    D += d
                    a_disp_next = a_prev * (1 - D) + a_next * D
                    a_disp = a_disp_next

                if k != 0:
                    K += k
                    a_vel_next = a_prev * (1 - K) + a_next * K                    
                    r_i = r_a(a_vel, cosmo, conf)
                    r_f = r_a(a_vel_next, cosmo, conf)
                    ray_cell_size, ray_mesh_shape = compute_ray_mesh(r_i, r_f, conf)
                    a_vel = a_vel_next
                    ray_cell_size_list.append(ray_cell_size)
                    ray_mesh_shape_list.append(ray_mesh_shape)
            return 
        integrate_ray(
            a_prev, a_next, cosmo, conf
        )

    # (number of K steps,) (number of K steps, 2)
    return cosmo.replace(ray_cell_size_list=jnp.asarray(ray_cell_size_list), ray_mesh_shape_list=jnp.asarray(ray_mesh_shape_list))