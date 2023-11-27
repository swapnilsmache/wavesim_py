import time
import numpy as np
from collections import defaultdict
from helmholtzbase import HelmholtzBase


class State(object):
    def __init__(self, base: HelmholtzBase):
        self.base = base
        self.init_norm = None
        self.subdomain_residuals = defaultdict(list)  # Initialize empty dict of lists
        self.full_residuals = []
        self.u_iter = defaultdict(list)  # Initialize empty dict of lists
        self.iterations = self.base.max_iterations
        self.should_terminate = False
        self.start_time = time.time()
        self.sim_time = 0

    def log_subdomain_residual(self, residual_s, j):
        """ Normalize subdomain residual wrt preconditioned source """
        self.subdomain_residuals[j].append((residual_s/self.init_norm).astype(np.float32))

    def log_full_residual(self, residual_f):
        """ Normalize full domain residual wrt preconditioned source """
        self.full_residuals.append(residual_f / self.init_norm)

    def log_u_iter(self, u, patch):
        """ Collect u_iter after rescaling and cropping to ROI"""
        u = np.sqrt(self.base.scaling[patch]) * u[self.base.crop2roi]  # Crop u to ROI and rescale

        # Collect u_iter 
        # mainly for plotting animation, so process accordingly:
        # 2D -- abs() for imshow; 3D -- abs() of 2D-slice in the middle for imshow
        if self.base.n_dims == 1:
            self.u_iter[patch].append(u)
        elif self.base.n_dims == 2:
            self.u_iter[patch].append(np.abs(u))
        elif self.base.n_dims == 3:
            self.u_iter[patch].append(np.abs(u[..., np.array([0, u.shape[-1]//2, -1])]))

    def next(self, i):
        """ Check termination conditions and to proceed to next iteration or not """
        if (self.full_residuals[i] <= self.base.threshold_residual
                or self.full_residuals[i] >= self.base.divergence_limit
                or i >= self.base.max_iterations - 1):
            print(f'Residual {self.full_residuals[i]:.2e}. '
                  f'Stopping at iteration {i + 1} ')
            self.should_terminate = True
            self.iterations = i+1
            self.sim_time = time.time() - self.start_time
            print('Simulation done (Time {} s)'.format(np.round(self.sim_time, 2)))

    def finalize(self, u):
        """ Rescale u and crop to ROI, and convert residual lists to arrays """
        for patch in self.base.domains_iterator:  # patch gives 3-element position tuple of subdomain (e.g., (0,0,0))
            current_patch = tuple([slice(patch[j]*(self.base.domain_size[j]-self.base.overlap[j]),
                                   patch[j]*(self.base.domain_size[j]-self.base.overlap[j])+self.base.domain_size[j])
                                   for j in range(self.base.n_dims)])
            u[current_patch] = np.sqrt(self.base.scaling[patch]) * u[current_patch]  # rescale u
        u = u[self.base.crop2roi]  # Crop u to ROI

        # convert residuals to arrays and reshape if needed
        self.subdomain_residuals = np.array(list(map(list, self.subdomain_residuals.values())))
        if self.subdomain_residuals.shape[0] < self.subdomain_residuals.shape[1]:
            self.subdomain_residuals = self.subdomain_residuals.T
        self.full_residuals = np.array(self.full_residuals)
        return u
