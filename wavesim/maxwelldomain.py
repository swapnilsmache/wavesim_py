import torch
from .domain import Domain
from .utilities import is_zero
from torch.cuda import empty_cache


class MaxwellDomain(Domain):
    """Represents a single domain of the simulation.

    The `Domain` object encapsulates all data that is stored on a single computation node 
    (e.g. a GPU or a node in a cluster), and provides methods to perform the basic operations 
    that the Wavesim algorithm needs.

    Note:
        MaxwellDomain has almost complete overlap with HelmholtzDomain except for the 
        propagator() function and having a 4th polarization dimension of size 3 in both slots.
        Eventually, the Domain class should be refactored, and Maxwell and HelmholtzDomain merged.
    """

    def __init__(self,
                 permittivity,
                 periodic: tuple[bool, bool, bool],
                 pixel_size: float = 0.25,
                 wavelength: float = None,
                 n_boundary: int = 0,
                 n_slots=2,
                 stand_alone=True,
                 Vwrap=None,
                 device=None,
                 debug=False):
        """Construct a domain object with the given permittivity and allocate memory.

        Note: the permittivity array is stored in one of the temporary memory slots and will be overwritten during
              processing. This means that no copy is kept (to save memory), and the data should not be used after
              calling this function.
        Note: all operations performed on this domain will use the same pytorch device and data type as the
              permittivity array.

        Args:
            permittivity: permittivity (n²) map. Must be a 3-dimensional array of complex float32 or float64.
                Its shape (n_x, n_y, n_z) is used to determine the size of the domain, and the device and datatype are
                used for all operations.
            periodic: tuple of three booleans indicating whether the domain is periodic in each dimension.            
            pixel_size: grid spacing (in wavelength units).
            wavelength: wavelength in micrometer (um).
            n_boundary: Number of pixels used for the boundary correction.
            n_slots: number of arrays used for storing the field and temporary data.
            stand_alone: if True, the domain performs shifting and scaling of the scattering potential (based on the
                permittivity of this domain alone). In this stand-alone mode, no wrapping corrections are applied,
                 making it equivalent to the original Wavesim algorithm.
                 Set to False when part of a multi-domain, where the all subdomains need to be considered together to
                 compute the shift and scale factors.
            Vwrap: optional wrapping matrix, when omitted and not in stand-alone mode, the matrix will be computed.
            device: 'cpu' to use the cpu, 'cuda' to distribute the simulation over all available cuda devices, 
                    'cuda:x' to use a specific cuda device, 
                    a list of strings, e.g., ['cuda:0', 'cuda:1'] to distribute the simulation over these 
                        devices in a round-robin fashion, or 
                    None, which is equivalent to 'cuda' if cuda devices are available, and 'cpu' if they are not.
            debug: set to True to return inverse_propagator_kernel as output.
         """

        if device is None or device == 'cuda':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if not torch.is_tensor(permittivity):
            permittivity = torch.tensor(permittivity, device=device)
        elif permittivity.device != device:
            permittivity.to(device)
        super().__init__(pixel_size, permittivity.shape, permittivity.device)

        # validate input arguments
        if n_slots < 2:
            raise ValueError("n_slots must be at least 2")
        if permittivity.ndim != 3 or not (
                permittivity.dtype == torch.complex64 or permittivity.dtype == torch.complex128):
            raise ValueError(
                f"Permittivity must be 3-dimensional and complex float32 or float64, not {permittivity.dtype}.")
        if any([n_boundary > 0.5 * self.shape[i] and not periodic[i] for i in range(3)]):
            raise ValueError(f"Domain boundary of {n_boundary} is too large for the given domain size {self.shape}")

        self._n_boundary = n_boundary
        self._Bscat = None
        self._periodic = periodic if n_boundary > 0 else \
            [True, True, True]  # allow manually disabling wrapping corrections by setting n_boundary=0
        self._source = None
        self._stand_alone = stand_alone
        self._debug = debug  # set to True to return inverse_propagator_kernel as output

        # allocate memory for the side pixels
        # note: at the moment, compute_corrections does not support in-place operations,
        # so this pre-allocated memory is not used.

        # edge_slices is a list of 6 pairs of slices, each pair corresponding to one of the six faces of the domain.
        def compute_slice(dd):
            d = dd // 2
            if dd % 2 == 0:
                return (slice(None),) * d + (slice(0, n_boundary),)
            else:
                return (slice(None),) * d + (slice(-n_boundary, None),)

        self.edge_slices = [compute_slice(dd) for dd in range(6)]
        self.edges = [
            None if self._periodic[0] else torch.zeros_like(permittivity[self.edge_slices[0]]),
            None if self._periodic[0] else torch.zeros_like(permittivity[self.edge_slices[1]]),
            None if self._periodic[1] else torch.zeros_like(permittivity[self.edge_slices[2]]),
            None if self._periodic[1] else torch.zeros_like(permittivity[self.edge_slices[3]]),
            None if self._periodic[2] else torch.zeros_like(permittivity[self.edge_slices[4]]),
            None if self._periodic[2] else torch.zeros_like(permittivity[self.edge_slices[5]]),
        ]

        # compute the un-scaled laplacian kernel and the un-scaled wrapping correction matrices
        # This kernel is given by -(px² + py² + pz²), with p_ the Fourier space coordinate
        # We temporarily store the kernel in `propagator_kernel`.
        # The shift and scale functions convert it to 1 / (scale·(L+shift)+1)
        # todo: convert to on-the-fly computation as in MATLAB code so that we don't need to store the kernel
        self.propagator_kernel = 0.0j
        for dim in range(3):
            self.propagator_kernel = self.propagator_kernel + self._laplace_kernel(dim)
        # self.propagator_kernel = None  # will be set in initialize_scale

        # compute the Fourier-space coordinates for the laplace operator. 
        # Used in the propagator() function.
        self._f = [self.coordinates_f(dim) for dim in range(3)]

        # allocate storage for temporary data, re-use the memory we got for the raw scattering potential
        # as one of the locations (which will be overwritten later)
        # self._x = [permittivity] + [torch.zeros_like(permittivity) for _ in range(n_slots - 1)]
        
        # modify storage for permittivity and temporary data to include the polarization dimension (size 3)
        self._x = [permittivity] + [torch.zeros_like(permittivity) for _ in range(n_slots - 1)]

        # compute n²·k₀² (the raw scattering potential)
        # also compute the bounding box holding the values of the scattering potential in the complex plane.
        # note: wavelength [pixels] = 1/self.pixel_size, so k=n·2π·self.pixel_size if wavelength is None
        if wavelength is None:
            permittivity.mul_(-(2.0 * torch.pi * self.pixel_size) ** 2)
        else:
            permittivity.mul_(-(2.0 * torch.pi / wavelength) ** 2)
        r_min, r_max = torch.aminmax(permittivity.real)
        i_min, i_max = torch.aminmax(permittivity.imag)
        self.V_bounds = torch.tensor((r_min, r_max, i_min, i_max))

        if stand_alone:
            # When in stand-alone mode, compute scaling factors now.
            self.Vwrap = [None, None, None]
            center = 0.5 * (r_min + r_max)  # + 0.5j * (i_min + i_max)
            V_norm = self.initialize_shift(center)
            self.initialize_scale(0.95j / V_norm)
        elif Vwrap is not None:
            # Use the provided wrapping matrices. This is used to ensure all subdomains use the same wrapping matrix
            self.Vwrap = [W.to(self.device) if W is not None else None for W in Vwrap]
        else:
            self.inverse_propagator_kernel = None  # self.propagator_kernel is the inverse propagator kernel
            # (memory efficient to store one instead of both)

            # Compute the wrapping correction matrices if none were provided
            # These matrices must be computed before initialize_scale, since they
            # affect the overall scaling.
            # place a point at -1,-1,-1 in slot 1 (which currently holds all zeros)
            # and then convolve the point with the inverse propagator kernel
            # we now have the wrap-around artefacts located at [:,-1,-1], [-1,:,-1] and [-1,-1,:]
            self._x[1][-1, -1, -1] = 1.0
            self.inverse_propagator(1, 1)
            self.Vwrap = [
                _make_wrap_matrix(self._x[1][:, -1, -1], n_boundary) if not self._periodic[0] else None,
                _make_wrap_matrix(self._x[1][-1, :, -1], n_boundary) if not self._periodic[1] else None,
                _make_wrap_matrix(self._x[1][-1, -1, :], n_boundary) if not self._periodic[2] else None,
            ]

        # # compute the norm of Vwrap. Worst case: just add all norms
        self.Vwrap_norm = sum([torch.linalg.norm(W, ord=2).item() for W in self.Vwrap if W is not None])

        # # modify storage for permittivity and temporary data to include the polarization dimension (size 3)
        # self._x[0] = self._x[0][..., None].tile(3,)
        # self._x[-1] = self._x[-1][..., None].tile(3,)
        self._x[0] = torch.cat((self._x[0][..., None], torch.zeros_like(self._x[0][..., None]), torch.zeros_like(self._x[0][..., None])), dim=-1)  # add polarization dimension
        self._x[1] = torch.cat((self._x[1][..., None], torch.zeros_like(self._x[1][..., None]), torch.zeros_like(self._x[1][..., None])), dim=-1)  # add polarization dimension
        # self._x = [permittivity[..., None].tile(3,)] + [torch.zeros_like(permittivity[..., None]).tile(3,) for _ in range(n_slots - 1)]
        # ## Shouldn't this happen on line 127? This might have been the cause of the error in the previous trial.

        print([x.shape for x in self._x])

        # Setup to iterate over domains only when the source is non-zero, 
        # or the norm of transfer corrections consistently increases
        self.mnum0 = [0.0] * 2  # store the last two values of the transfer correction norm for 1st medium call
        self.mnum1 = [0.0] * 2  # ... for 2nd medium call
        self.counter = 0  # counter to keep track of the number of iterations with increasing transfer correction norm
        self.active = True  # flag to indicate if the domain is active in the iteration

        empty_cache()  # free up memory before going to run_algorithm

    # Functions implementing the domain interface
    # add_source()
    # clear()
    # get()
    # inner_product()
    # medium()
    # mix()
    # propagator()
    # set_source()
    def add_source(self, slot: int, weight: float):
        """Adds the source term to the data in the specified slot."""
        if self._source is not None:
            if self.active:
                torch.add(self._x[slot], self._source, out=self._x[slot], alpha=weight)

    def clear(self, slot: int):
        """Clears the data in the specified slot"""
        self._x[slot].zero_()

    def get(self, slot: int, copy=False):
        """Returns the data in the specified slot.

        :param slot: slot from which to return the data
        :param copy: if True, returns a copy of the data. Otherwise, may return the original data possible.
                     Note that this data may be overwritten by the next call to domain.
        """
        data = self._x[slot]
        return data.detach().clone() if copy else data

    def set(self, slot: int, data):
        """Copy the date into the specified slot"""
        self._x[slot].copy_(data)

    def inner_product(self, slot_a: int, slot_b: int):
        """Computes the inner product of two data vectors

        Note: 
            The vectors may be represented as multidimensional arrays,
            but these arrays must be contiguous for this operation to work.
            Although it would be possible to use flatten(), this would create a
            copy when the array is not contiguous, causing a hidden performance hit.
        """
        retval = torch.vdot(self._x[slot_a].view(-1), self._x[slot_b].view(-1)).item()
        return retval if slot_a != slot_b else retval.real  # remove small imaginary part if present

    def medium(self, slot_in: int, slot_out: int, mnum = None):
        """Applies the operator 1-Vscat.

        Note: 
            Does not apply the wrapping correction. When part of a multi-domain,
            the wrapping correction is applied by the medium() function of the multi-domain object
            and this function should not be called directly.
        """
        if self.active:
            torch.mul(self._Bscat[..., None], self._x[slot_in], out=self._x[slot_out])

    def mix(self, weight_a: float, slot_a: int, weight_b: float, slot_b: int, slot_out: int):
        """Mixes two data arrays and stores the result in the specified slot"""
        if self.active:
            a = self._x[slot_a]
            b = self._x[slot_b]
            out = self._x[slot_out]
            if weight_a == 1.0:
                torch.add(a, b, alpha=weight_b, out=out)
            elif weight_a == 0.0:
                torch.mul(b, weight_b, out=out)
            elif weight_b == 1.0:
                torch.add(b, a, alpha=weight_a, out=out)
            elif weight_b == 0.0:
                torch.mul(a, weight_a, out=out)
            elif weight_a + weight_b == 1.0:
                torch.lerp(a, b, weight_b, out=out)
            elif slot_a == slot_out:
                a.mul_(weight_a)
                a.add_(b, alpha=weight_b)
            else:
                torch.mul(b, weight_b, out=out)
                out.add_(a, alpha=weight_a)

    def propagator(self, slot_in: int, slot_out: int):
        """Applies the operator (L+1)^-1 x.
        Steps:
            x = F(x)
            y = p⊗p·x·c / (c·k0^2 + 1)
            x = x + y
            x = x / (c[p² + k0²] + 1)
            x = F^-1(x)
        """
        # 1. F(x)
        # 2. p⊗p·F(x)·c / (c·k0^2 + 1)
        # 3. F(x) + p⊗p·F(x)·c / (c·k0^2 + 1)
        # 4. (F(x) + p⊗p·F(x)·c / (c·k0^2 + 1)) / (c[p² + k0²] + 1)
        # 5. F^-1 ( (F(x) + p⊗p·F(x)·c / (c·k0^2 + 1)) / (c[p² + k0²] + 1) )

        # 3. F(E) + c∇∇·F(E) / (c·k0^2 + 1)
        # 4. (F(E) + c∇∇·F(E) / (c·k0^2 + 1)) / (c[p^2 + k0^2] + 1)
        # 5. F^-1 ( (F(E) + c∇∇·F(E) / (c·k0^2 + 1)) / (c[p^2 + k0^2] + 1) )

        if self.active:
            for d in range(3):
                torch.fft.fftn(self._x[slot_in][..., d], out=self._x[slot_out][..., d])

            div = self.scale * sum(self._f[d] * self._x[slot_out][..., d] for d in range(3))/(self.scale*self.shift+1.0)

            for d in range(3):
                self._x[slot_out][..., d].add_(self._f[d].mul(div))
                self._x[slot_out][..., d].mul_(self.propagator_kernel)

            for d in range(3):
                torch.fft.ifftn(self._x[slot_out][..., d], out=self._x[slot_out][..., d])

        # if self.active:
            # for d in range(3):
            #     torch.fft.fftn(self._x[slot_in][..., d], out=self._x[slot_out][..., d])

            # div = sum(self._f[d] * self._x[slot_out][..., d] for d in range(3))  # div
            # div.div_(sum(self._f[d]**2 for d in range(3)))  # div / (n² k₀²)
            # # f2 = 0.0j
            # # for dim in range(3):
            # #     f2 = f2 + self._laplace_kernel(dim)
            # # div.div_(f2)  # div / (n² k₀²)
            # grad_div = torch.zeros_like(self._x[slot_out])
            # for d in range(3):
            #     grad_div[..., d] = self._f[d] * div  # grad div
            #     grad_div[0, 0, 0, d] = self._x[slot_out][0, 0, 0, d]

            # self._x[slot_out].sub_(grad_div)
            # for d in range(3):
            #     self._x[slot_out][..., d].mul_(self.propagator_kernel)

            # grad_div.div_(self.scale * self.shift + 1.0)
            # self._x[slot_out].add_(grad_div)

            # for d in range(3):
            #     torch.fft.ifftn(self._x[slot_out][..., d], out=self._x[slot_out][..., d])

    def inverse_propagator(self, slot_in: int, slot_out: int):
        """Applies the operator (L+1) x .

        This operation is not needed for the Wavesim algorithm, but is provided for testing purposes,
        and can be used to evaluate the residue of the solution.
        """
        # todo: convert to on-the-fly computation
        torch.fft.fftn(self._x[slot_in], out=self._x[slot_out])
        if self.inverse_propagator_kernel is None:
            # self.propagator_kernel is the inverse propagator kernel (memory efficient to store one instead of both)
            self._x[slot_out].mul_(self.propagator_kernel)
        else:
            self._x[slot_out].mul_(self.inverse_propagator_kernel)
        torch.fft.ifftn(self._x[slot_out], out=self._x[slot_out])

    def set_source(self, source):
        """Sets the source term for this domain."""
        self._source = None
        if source is None or is_zero(source):
            return

        source = source.to(self.device)
        if source.is_sparse:
            source = source.coalesce()
            if len(source.indices()) == 0:
                return

        self._source = source.to(self.device, self._x[0].dtype)

    def initialize_shift(self, shift) -> float:
        """Shifts the scattering potential and propagator kernel, then returns the norm of the shifted operator."""
        self.propagator_kernel.add_(shift)
        self._x[0].add_(-shift)  # currently holds the scattering potential
        self.shift = shift
        return self._x[0].view(-1).abs().max().item()

    def initialize_scale(self, scale: complex):
        """Scales all operators.

        Computes Bscat (from the temporary storage 0), the propagator kernel (from the temporary value in
        propagator_kernel), and scales Vwrap.

        Attributes:
            scale: Scaling factor of the problem. Its magnitude is chosen such that the operator 
                   V = scale · (the scattering potential + the wrapping correction) has norm < 1. 
                   The complex argument is chosen such that L+V is accretive.
        """

        # B = 1 - scale·(n² k₀² - shift). Scaling and shifting was already applied. 1-... not yet
        self.scale = scale
        self._Bscat = self._x[0].clone()
        self._Bscat.mul_(-scale)
        self._Bscat.add_(1.0)
        # self._Bscat = 1.0 - scale * self._x[0]

        # kernel = 1 / (scale·(L + shift) + 1). Shifting was already applied. scaling, +1 and reciprocal not yet
        self.propagator_kernel.multiply_(scale)
        self.propagator_kernel.add_(1.0)
        if self._debug:
            self.inverse_propagator_kernel = self.propagator_kernel.clone()
        self.propagator_kernel.reciprocal_()

        if self.Vwrap is not None:
            self.Vwrap = [scale * W if W is not None else None for W in self.Vwrap]

    def compute_corrections(self, slot_in: int):
        """Computes the edge corrections by multiplying the first and last pixels of each line with 
        the Vwrap matrix.

        The corrections are stored in self.edges.
        TODO: re-use this memory
        """
        for edge in range(6):
            axes = [1, ] if edge % 2 == 0 else [0, ]
            dim = edge // 2
            if self.Vwrap[dim] is None:
                continue

            view = torch.moveaxis(self.edges[edge], dim, 0)
            torch.tensordot(a=self.Vwrap[dim], b=self._x[slot_in][self.edge_slices[edge]], dims=(axes, [dim, ]),
                            out=view)

        return self.edges

    def apply_corrections(self, wrap_corrections, transfer_corrections, slot: int):
        """Apply wrapping/transfer corrections

        Transfer corrections correspond to a contribution from neighboring domains, and are added
        to the current domain. Wrap corrections correct for the periodicity of the fft. They are 
        subtracted from the domain. In this case, there is an additional factor of -1 because 
        this function is called from `medium`, which applies 1-V instead of V.
        Therefore, transfer corrections are now subtracted, and wrap corrections are added.

        :param slot: slot index for the data to which the corrections are applied. Operation is always in-place
        :param wrap_corrections: list of 6 corrections for wrap-around (may contain None for periodic boundary)
        :param transfer_corrections: list of 6 corrections coming from neighboring segments (may contain None for
                                     end of domain)
        """
        for edge in range(6):
            if wrap_corrections[edge] is not None and transfer_corrections[edge] is None:
                self._x[slot][self.edge_slices[edge]] += wrap_corrections[edge]
            elif transfer_corrections[edge] is not None and wrap_corrections[edge] is None:
                self._x[slot][self.edge_slices[edge]] -= transfer_corrections[edge]
            elif transfer_corrections[edge] is not None and wrap_corrections[edge] is not None:
                self._x[slot][self.edge_slices[edge]] += wrap_corrections[edge] - transfer_corrections[edge].to(
                    self.device)
            else:
                pass

    def _laplace_kernel(self, dim):
        """Compute the Fourier-domain kernel for the Laplace operator in the given dimension"""

        # original way (introduces wrapping artifacts in the kernel)
        # return -self.coordinates_f(dim) ** 2

        # new way: uses exact Laplace kernel in real space, and returns Fourier transform of that
        x = self.coordinates(dim, 'periodic')
        if x.numel() == 1:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        x = x * torch.pi / self.pixel_size
        c = torch.cos(x)
        s = torch.sin(x)
        x_kernel = 2.0 * c / x ** 2 - 2.0 * s / x ** 3 + s / x
        x_kernel[0, 0, 0] = 1.0 / 3.0  # remove singularity at x=0
        x_kernel *= -torch.pi ** 2 / self.pixel_size ** 2
        f_kernel = torch.fft.fftn(x_kernel).to(torch.complex64)
        return -f_kernel.real

    def create_empty_vdot(self):
        """Creates an empty tensor to force the allocation of memory for the Vdot tensor.

        This tensor is always 8.1 MiB, irrespective of the domain size.
        """
        self.empty_vdot = torch.empty((1024, 2500), dtype=self._x[0].dtype, device=self.device)
        del self.empty_vdot  # frees up the memory, but keeps the segment, so it can be re-used


def _make_wrap_matrix(L_kernel, n_boundary):
    """ Make the matrices for the wrapping correction

    :param L_kernel: the kernel for the laplace operator
    :param n_boundary: the size of the correction matrix
    :return: the wrapping correction matrix

    Note: the matrices need may not be identical for the different dimensions if the sizes are different
    Note: uses the temporary storage slot 1 for the intermediate results
    """

    # define a single point source at (0,0,0) and compute the (wrapped) convolution
    # with the forward kernel (L+1)
    kernel_section = L_kernel.real.ravel()

    # construct a non-cyclic convolution matrix that computes the wrapping artifacts only
    wrap_matrix = torch.zeros((n_boundary, n_boundary), dtype=kernel_section.dtype, device=kernel_section.device)
    for r in range(n_boundary):
        size = r + 1
        wrap_matrix[r, :] = kernel_section[n_boundary - size:2 * n_boundary - size]
    return wrap_matrix
