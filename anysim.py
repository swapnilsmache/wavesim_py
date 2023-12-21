import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from helmholtzbase import HelmholtzBase
from state import State


def iterate(base: HelmholtzBase):
    """ AnySim update
    :param base: Helmholtz base parameters
    :return: computed field u, state object """

    u = np.zeros_like(base.s, dtype=np.complex64)  # field u, initialize with 0s
    restrict, extend = domain_decomp_operators(base)  # Construct restriction and extension operators
    state = State(base)
    state.init_norm = norm(np.sum(np.array([(map_domain(base.medium_operators[patch](base.propagator(map_domain(
        base.s, restrict, patch), base.scaling[patch])), extend, patch)) for patch in base.domains_iterator]), axis=0))

    # Empty dicts of lists to store patch-wise source (s) and field (u)
    s_dict = defaultdict(list)
    u_dict = defaultdict(list)
    for patch in base.domains_iterator:
        # restrict full-domain source s to the patch subdomain, and apply scaling for that subdomain
        s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * map_domain(base.s, restrict, patch)
        # restrict full-domain field u to the patch subdomain
        u_dict[patch] = map_domain(u, restrict, patch)

    for i in range(base.max_iterations):
        residual = 0
        for patch in base.domains_iterator:  # patch gives the 3-element position tuple of subdomain
            print(f'Iteration {i + 1}, sub-domain {patch}. ', end='\r')

            t1 = base.medium_operators[patch](u_dict[patch]) + s_dict[patch]  # B(u) + s
            t1 = t1 - base.transfer_correction(u_dict, patch)  # Add transfer_correction w/ previous &// next subdomain
            t1 = base.propagator(t1, base.scaling[patch])  # (L+1)^-1 t1
            t1 = base.medium_operators[patch](u_dict[patch] - t1)  # B(u - t1). subdomain residual

            state.log_subdomain_residual(norm(t1), patch)  # log residual for current subdomain

            u_dict[patch] = u_dict[patch] - (base.alpha * t1)  # update subdomain u

            # find the slice of full domain u and update
            patch_slice = base.patch_slice(patch)
            u[patch_slice] = u_dict[patch]

            state.log_u_iter(u, patch)  # collect u updates (store separately subdomain-wise)
            residual += map_domain(t1, extend, patch)  # add up all subdomain residuals
        state.log_full_residual(norm(residual))  # log residual for entire domain
        state.next(i)  # Check termination conditions
        if state.should_terminate:  # Proceed to next iteration or not
            break
    # return u and u_iter cropped to roi, residual arrays, and state object with information on run
    return state.finalize(u), state


def domain_decomp_operators(base):
    """ Construct restriction and extension operators """
    restrict = [[] for _ in range(base.n_dims)]
    extend = [[] for _ in range(base.n_dims)]

    if base.total_domains == 1:
        [restrict[dim].append(1.) for dim in range(base.n_dims)]
        [extend[dim].append(1.) for dim in range(base.n_dims)]
    else:
        ones = np.eye(base.domain_size[0])
        restrict0_ = []
        [restrict0_.append(np.zeros((base.domain_size[dim], base.n_ext[dim]))) 
            for dim in range(base.n_dims)]
        for dim in range(base.n_dims):
            for patch in range(base.n_domains[dim]):
                restrict_mid_ = restrict0_[dim].copy()
                restrict_mid_[:, slice(patch * base.domain_size[dim],
                                       patch * base.domain_size[dim] + base.domain_size[dim])] = ones
                restrict[dim].append(restrict_mid_.T)
                extend[dim].append(restrict_mid_)
    return restrict, extend


def map_domain(x, mapping_operator, patch):
    """ Map x to extended domain or restricted subdomain """
    n_dims = np.squeeze(x).ndim
    for dim in range(n_dims):  # For applying in every dimension
        x = np.moveaxis(x, dim, -1)  # Transpose
        x = np.dot(x, mapping_operator[dim][patch[dim]])  # Apply (appropriate) mapping operator
        x = np.moveaxis(x, -1, dim)  # Transpose back
    return x.astype(np.complex64)
