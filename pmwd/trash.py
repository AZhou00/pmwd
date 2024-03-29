def gather_eta(ray_pos_ip, ray_pmid, defl_2D_smth, ray_cell_size, conf):
    """Compute the kick operator on the ray poasitions.

    Args:
        ray_pos_ip (_type_): shape (ray_num * 2,), flattened position array
        defl_2D_smth (_type_): shape (ray_mesh_shape[0], ray_mesh_shape[1], 2), smoothed deflection field
        ray_pmid (_type_): shape (ray_num, 2), ray.pmid
        ray_cell_size (_type_): _description_
        conf (_type_): _description_

    Returns:
        _type_: _description_
    """
    ray_pos_ip = ray_pos_ip.reshape(-1, 2)
    ray_mesh_shape = defl_2D_smth.shape[0:2]

    deta = _gather_rt(
        pmid = jnp.zeros_like(ray_pmid).astype(conf.pmid_dtype), 
        disp = ray_pos_ip, 
        conf = conf, 
        mesh = defl_2D_smth, 
        # val is 2D: deta in x and y
        val = jnp.zeros((conf.ray_num,2)), 
        offset = ray_mesh_center(ray_cell_size, ray_mesh_shape), 
        ray_cell_size = ray_cell_size,
        ray_mesh_shape = ray_mesh_shape
        )
    # visual_lensing(ptcl, defl_2D, defl_2D_smth, coord3D, conf, z, chi_i, chi_f, ray_mesh_shape, ray_cell_size)
    
    # print('deta', deta.shape) # (ray_num, 2)
    
    return deta.reshape(-1)


def lensing(a_i, a_f, a_c, ptcl, ray, grad_phi3D, cosmo, conf):
    """    # Reshaping here follows this pattern below
        # A = jnp.arange(12).reshape(3,2,2)
        # print(A.shape)
        # print(A)
        # Ax = A[:,:,0]
        # Ay = A[:,:,1]
        # print()
        # print('Ax \n', Ax)
        # print('Ay \n', Ay)
        # Ax = Ax.reshape(-1)
        # Ay = Ay.reshape(-1)
        # print()
        # print('Ax \n', Ax)
        # print('Ay \n', Ay)
        # Ax = Ax.reshape(-1,2)
        # Ay = Ay.reshape(-1,2)
        # print()
        # print('Ax \n', Ax)
        # print('Ay \n', Ay)
        # A_new = jnp.stack((Ax,Ay), axis=-1)
        # print(jnp.allclose(A, A_new))

        # deta, f_jvp = linearize(f, ray.pos_ip() )
        # deta = deta.reshape(-1,2)
        # dB = f_jvp(ray.A)
        # print('dB', dB.shape) # (ray_num, 2)
    """
    
    r_i = distance_ad(a_i, cosmo, conf)
    r_f = distance_ad(a_f, cosmo, conf)
    ray_cell_size, ray_mesh_shape = compute_ray_mesh(r_i, r_f, conf)
    defl_2D_smth = deflection_field(a_i, a_f, a_c, ptcl, grad_phi3D, ray_cell_size, ray_mesh_shape, cosmo, conf)
    # print defl_2D_smth.shape # (ray_mesh_shape[0], ray_mesh_shape[1], 2)
    f = partial(
        gather_eta,
        ray_pmid=ray.pmid,
        defl_2D_smth=defl_2D_smth,
        ray_cell_size=ray_cell_size,
        conf=conf,
    )
    
    # first/second column of the distortion matrix (dtheta_n1_dtheta_0_x/y), 
    # flattened, shape = (ray_num * 2)
    
    # ray.pos_ip() shape = (ray_num, 2)
    # A.
    A0_x = ray.A[:,:,0].reshape(-1) 
    A0_y = ray.A[:,:,1].reshape(-1)
    # A0_x = ray.A[:,0,:].reshape(-1) 
    # A0_y = ray.A[:,1,:].reshape(-1)
    
    # ray positions, flattened, shape = (ray_num * 2)
    primals = ray.pos_ip().reshape(-1)
    
    # useful notes on linearize https://github.com/google/jax/issues/526
    # gradient, reshape results to (ray_num, 2)
    deta, f_jvp = linearize(f, primals) # deta' shape = 
    deta = deta.reshape(-1,2)
    dB_x = f_jvp(A0_x).reshape(-1,2)
    dB_y = f_jvp(A0_y).reshape(-1,2)
    # integers -> float0, others use jnp.zeros.
    # primals = (x,y,z)
    # f_jvp(x_v, jnp.zeros_like(y), jnp.zeros_like(z))
    
    # # stack two columns vector side by side
    dB = jnp.stack([dB_x, dB_y], axis=-1)
    
    return deta, dB