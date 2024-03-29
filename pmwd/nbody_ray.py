from functools import partial

from jax import value_and_grad, jit, vjp, custom_vjp
import jax.numpy as jnp

from pmwd.boltzmann import growth, distance_cm, distance_ad
from pmwd.gravity import gravity
from pmwd.lensing import grad_phi, lensing
from pmwd.nbody import nbody_init, nbody_step

def force_ray(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf):
    """deta on rays."""
    deta, dB = lensing(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf)
    return ray.replace(deta=deta, dB=dB)

def kick_ray(ray):
    """Kick."""
    eta = ray.eta + ray.deta
    B = ray.B + ray.dB
    return ray.replace(eta=eta, B=B)

def drift_ray(a_vel, a_prev, a_next, ray, cosmo, conf):
    """
    Drift.
    factor = (chi_{n+1}-\chi_n)/r(\chi_{n+1/2})^2/c
    """
    factor = distance_cm(a_next, cosmo, conf) - distance_cm(a_prev, cosmo, conf)
    factor /= distance_ad(a_vel, cosmo, conf) ** 2
    factor /= conf.c
    
    disp = ray.disp + factor * ray.eta 
    A = ray.A + factor * ray.B
    
    print("max drift [arcmin]", jnp.max(jnp.abs(ray.eta * factor) * 3437.75))
    return ray.replace(disp=disp, A=A)

def integrate_ray(a_prev, a_next, ptcl, ray, cosmo, conf):
    """Symplectic integration for one step."""
    print('------------------')
    D = K = 0
    a_disp = a_vel = a_prev
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ray = drift_ray(a_vel, a_disp, a_disp_next, ray, cosmo, conf)
            a_disp = a_disp_next

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            a_c = (a_vel + a_vel_next) / 2
            grad_phi3D = grad_phi(a_c, ptcl, cosmo, conf) 
            ray = force_ray(a_vel, a_vel_next, a_c, ptcl, ray, grad_phi3D, cosmo, conf)
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
    A,B = ray.A, ray.B
    # observables have shape (ray_num,)
    # kappa, gamma1, gamma2, omega = obsvbl_ray
    
    C1 = (A[...,0,0] + A[...,1,1]) / 2
    C2 = (A[...,0,1] - A[...,1,0]) / 2
    C3 = (A[...,1,1] - A[...,0,0]) / 2
    C4 = (A[...,0,1] + A[...,1,0]) / 2
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

def nbody_ray_step(a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf):
    ray = integrate_ray(a_prev, a_next, ptcl, ray, cosmo, conf)
    obsvbl_ray = observe(ray, obsvbl_ray, cosmo, conf)
    return ray, obsvbl_ray

def nbody_ray(ptcl, ray, obsvbl, obsvbl_ray, cosmo, conf, reverse=True):
    """
    N-body time integration with ray tracing updates.
    ray tracing nbody integration goes backward by default
    """
    # a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    a_nbody = conf.a_nbody_ray  # in reverse order up the maximum source redshift

    # initialize the acceleration to ptcl does not do anything else.
    ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)
    # does nothing right now
    ray, obsvbl_ray = nbody_ray_init(a_nbody[0], ray, obsvbl_ray, cosmo, conf)
    # initialize the first kick operator,
    # which integrates up to the first half step k = conf.symp_splits[0][1]
    # a_i = a_nbody[0]
    # a_f = a_nbody[0] * (1 - conf.symp_splits[0][1])
    # a_f += a_nbody[1] * conf.symp_splits[0][1]
    # ray = force_ray(a_i, a_f, ptcl, ray, cosmo, conf)
    
    # nbody integration
    for a_prev, a_next in zip(a_nbody[:-1], a_nbody[1:]):
        ray, obsvbl_ray = nbody_ray_step(a_prev, a_next, ptcl, ray, obsvbl_ray, cosmo, conf)
        ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
        
    return ptcl, ray, obsvbl, obsvbl_ray
