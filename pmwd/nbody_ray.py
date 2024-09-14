from functools import partial

from jax import value_and_grad, jit, vjp, custom_vjp
import jax.numpy as jnp

from pmwd.boltzmann import growth, chi_a, r_a
from pmwd.lensing import grad_phi, lensing
from pmwd.nbody import nbody_init, nbody_step


def force_ray(a_i, a_f, a_c, ray, grad_phi3D, cosmo, conf, ray_mesh_params):
    """deta on rays."""
    deta, dB = lensing(a_i, a_f, a_c, ray, grad_phi3D, cosmo, conf, ray_mesh_params)
    return ray.replace(deta=deta, dB=dB)


def kick_ray(ray):
    """Kick."""
    eta = ray.eta + ray.deta
    B = ray.B + ray.dB
    return ray.replace(eta=eta, B=B)


def drift_ray(a_vel, a_prev, a_next, ray, cosmo, conf, ptcl):
    """
    Drift.
    factor = (chi_{n+1}-\chi_n)/r(\chi_{n+1/2})^2/c
    """
    factor = chi_a(a_next, cosmo, conf) - chi_a(a_prev, cosmo, conf)
    factor /= r_a(a_vel, cosmo, conf) ** 2
    factor /= conf.c

    # # point mass lens --------------------------------
    # chi_l = 770 #653.606 # in [L], i.e., Mpc
    # chi_s = chi_a(a_next, cosmo, conf)
    # if chi_s > chi_l:
    #     print( "chi_l", chi_l, "chi_s", chi_s,)
    #     chi_s_prev = chi_a(a_prev, cosmo, conf)
    #     M = cosmo.ptcl_mass * conf.ptcl_num * conf.mass_rescale #(conf.mesh_size / conf.ptcl_num) *
    #     visual_ray_point_mass(
    #         theta0=ray.pos_0(),
    #         theta_prev=ray.pos_ip(),
    #         theta_s=ray.pos_ip() + factor * ray.eta,
    #         chi_l=chi_l,
    #         chi_s_prev=chi_s_prev,
    #         chi_s=chi_s,
    #         M=M,
    #         conf=conf,
    #         ptcl=ptcl
    #         )
    # print("max drift [arcmin]", jnp.max(jnp.abs(ray.eta * factor) * 3437.75))
    # # ------------------------------------------------

    disp = ray.disp + factor * ray.eta
    A = ray.A + factor * ray.B
    return ray.replace(disp=disp, A=A)


def integrate_ray(a_prev, a_next, ptcl, ray, cosmo, conf, ray_mesh_params):
    """Symplectic integration for one step."""
    # print('------------------')
    D = K = 0
    a_disp = a_vel = a_prev
    
    # potential for the ray evaluated at half step
    # a dependency only affects SO
    if conf.so_type is not None:
        grad_phi3D = grad_phi((a_prev+a_next)/2, ptcl, cosmo, conf)
    else:
        # ignore first entry and statically jit it
        grad_phi3D = grad_phi(0, ptcl, cosmo, conf)

    # KDK
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ray = drift_ray(a_vel, a_disp, a_disp_next, ray, cosmo, conf, ptcl)
            # a_disp = a_disp_next

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            a_c = (a_vel + a_vel_next) / 2
            # a_c = a_next
            ray = force_ray(a_vel, a_vel_next, a_c, ray, grad_phi3D, cosmo, conf, ray_mesh_params)
            ray = kick_ray(ray)
            a_vel = a_vel_next
    return ray


def nbody_ray_init(a, ray, obsvbl_ray, cosmo, conf):
    return ray, obsvbl_ray


def observe(ray, obsvbl_ray, cosmo, conf):
    """
    n(z) can come from redshift. 
    can add observable dependent n(z) later
    
    C_1 & = \frac{A_{11} + A_{22}}{2} \,, \\
    C_2 & = \frac{A_{12} - A_{21}}{2} \,, \\
    C_3 & = \frac{A_{22} - A_{11}}{2} \,, \\
    C_4 & = \frac{A_{12} + A_{21}}{2} \,. \\
    \kappa   & = 1 - C_1\,,               \\
    \omega   & = \frac{C_2}{1-\kappa} \,, \\
    \gamma_1 & = C_3 + C_4~ \omega \,,    \\
    \gamma_2 & = - C_4 + C_3~ \omega \,,  \\
    """
    # A & B have shape (ray_num, 2, 2)
    A, B = ray.A, ray.B
    # observables have shape (ray_num,)
    # kappa, gamma1, gamma2, omega = obsvbl_ray

    C1 = (A[..., 0, 0] + A[..., 1, 1]) / 2
    C2 = (A[..., 0, 1] - A[..., 1, 0]) / 2
    C3 = (A[..., 1, 1] - A[..., 0, 0]) / 2
    C4 = (A[..., 0, 1] + A[..., 1, 0]) / 2
    kappa = 1 - C1
    omega = C2 / (1 - kappa)
    gamma1 = C3 + C4 * omega
    gamma2 = -C4 + C3 * omega

    # n(z)
    kappa *= 1
    omega *= 1
    gamma1 *= 1
    gamma2 *= 1

    obsvbl_ray = kappa, gamma1, gamma2, omega
    return obsvbl_ray

@partial(jit, static_argnums=(7, 8, 9))
def nbody_ray_step(a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf, ray_cell_size, ray_mesh_shape_x, ray_mesh_shape_y):
    ray = integrate_ray(a_prev, a_next, ptcl, ray, cosmo, conf, (ray_cell_size, (ray_mesh_shape_x, ray_mesh_shape_y)))
    obsvbl_ray = observe(ray, obsvbl_ray, cosmo, conf)
    return ray, obsvbl_ray


def nbody_ray(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf, ray_mesh_params, static=False):
    """
    N-body time integration with ray tracing updates.
    ray tracing nbody integration goes backward by default
    """
    a_nbody = conf.a_nbody_rt  # in reverse order up the maximum source redshift
    
    if not static:
        # initialize the acceleration to ptcl does not do anything else.
        ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)

    # does nothing right now
    ray, obsvbl_ray = nbody_ray_init(a_nbody[0], ray, obsvbl_ray, cosmo, conf)

    for i, (a_prev, a_next) in enumerate(zip(a_nbody[:-1], a_nbody[1:])):
        ray_cell_size_list, ray_mesh_shape_list = ray_mesh_params
        ray_cell_size, (ray_mesh_shape_x, ray_mesh_shape_y) = ray_cell_size_list[i*2+1], ray_mesh_shape_list[i*2+1]        
        ray, obsvbl_ray = nbody_ray_step(
            a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf, ray_cell_size, ray_mesh_shape_x, ray_mesh_shape_y
        )
        if not static:
            ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
    
    return ptcl, ray, obsvbl, obsvbl_ray