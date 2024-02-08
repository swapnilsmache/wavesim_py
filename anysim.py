import torch
import numpy as np
from torch.linalg import norm
from collections import defaultdict

from helmholtzbase import HelmholtzBase
from state import State


def iterate(base: HelmholtzBase):
    """ AnySim update
    :param base: Helmholtz base parameters
    :return: u (computed field), state (object) """

    # u = np.zeros_like(base.s, dtype=np.complex64)  # field u, initialize with 0s
    u = torch.zeros_like(base.s, device=base.device, dtype=torch.complex64)
    restrict, extend = domain_decomp_operators(base)  # Construct restriction and extension operators
    state = State(base)
    # list of preconditioner i.e. medium(propagator()) applied to source term in each subdomain patch
    norm_patch = [(map_domain(base.medium_operators[patch](base.propagator_operators[patch](
        map_domain(base.s, restrict, patch))), extend, patch)) for patch in base.domains_iterator]
    # initial norm for residual computation. Summing up all patches of medium(propagator(source)) and taking norm
    state.init_norm = norm(sum(norm_patch))

    # Empty dicts of lists to store patch-wise source (s) and field (u)
    s_dict = defaultdict(list)
    u_dict = defaultdict(list)
    ut_dict = defaultdict(list)
    for patch in base.domains_iterator:
        # restrict full-domain source s to the patch subdomain, and apply scaling for that subdomain
        s_dict[patch] = 1j * np.sqrt(base.scaling[patch]) * map_domain(base.s, restrict, patch)
        u_dict[patch] = map_domain(u, restrict, patch)  # restrict full-domain field u to the subdomain patch

    for i in range(base.max_iterations):
        print(f'Iteration {i + 1}', end='\r')
        residual = 0
        t_dict = base.medium(u_dict, s_dict)  # B(u) + s
        t_dict = base.transfer(u_dict, t_dict)
        t_dict = base.propagator(t_dict)  # (L+1)^-1 t
        for patch in base.domains_iterator:  # patch gives the 3-element position tuple of subdomain
            ut_dict[patch] = u_dict[patch] - t_dict[patch] # (u-t)
        t_dict = base.medium(ut_dict)  # B(u - t). subdomain residual
        t_dict = base.transfer(ut_dict, t_dict)
        for patch in base.domains_iterator:  # patch gives the 3-element position tuple of subdomain
            state.log_subdomain_residual(norm(t_dict[patch]), patch)  # log residual for current subdomain

            u_dict[patch] = u_dict[patch] - (base.alpha * t_dict[patch])  # update subdomain u

            # find the slice of full domain u and update
            patch_slice = base.patch_slice(patch)
            u[patch_slice] = u_dict[patch]

            # state.log_u_iter(u, patch)  # collect u updates (store separately subdomain-wise)
            residual += map_domain(t_dict[patch], extend, patch)  # add up all subdomain residuals

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
        # ones = np.eye(base.domain_size[0])
        ones = torch.eye(base.domain_size[0], dtype=torch.complex64, device=base.device)
        restrict0_ = []
        n_ext = base.n_roi + base.boundary_pre + base.boundary_post
        # [restrict0_.append(np.zeros((base.domain_size[dim], n_ext[dim]))) 
        #     for dim in range(base.n_dims)]
        [restrict0_.append(torch.zeros((base.domain_size[dim], n_ext[dim]), dtype=torch.complex64, device=base.device)) for dim in range(base.n_dims)]
        for dim in range(base.n_dims):
            for patch in range(base.n_domains[dim]):
                restrict_mid_ = restrict0_[dim].clone()
                restrict_mid_[:, slice(patch * base.domain_size[dim],
                                       patch * base.domain_size[dim] + base.domain_size[dim])] = ones
                restrict[dim].append(restrict_mid_.T)
                extend[dim].append(restrict_mid_)
    return restrict, extend


def map_domain(x, mapping_operator, patch):
    """ Map x to extended domain or restricted subdomain """
    if isinstance(mapping_operator[0][0], float):
        pass
    else:
        n_dims = torch.squeeze(x).ndim
        for dim in range(n_dims):  # For applying in every dimension
            x = torch.moveaxis(x, dim, -1)  # Transpose
            # x = np.dot(x, mapping_operator[dim][patch[dim]])  # Apply (appropriate) mapping operator
            x = torch.tensordot(x, mapping_operator[dim][patch[dim]], ((-1,),(0,))) # Apply (appropriate) mapping operator
            x = torch.moveaxis(x, -1, dim)  # Transpose back
    return x#.astype(np.complex64)
