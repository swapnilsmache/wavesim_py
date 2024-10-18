import pytest
import torch
from wavesim.maxwelldomain import MaxwellDomain
from . import allclose, random_vector, random_refractive_index, dtype

""" Performs a set of basic consistency checks for the MaxwellDomain class. """


def construct_domain(n_size, n_boundary, periodic=(False, False, True)):
    """ Construct a domain """
    torch.manual_seed(12345)
    n = random_refractive_index(n_size)
    return MaxwellDomain(permittivity=n, periodic=periodic, n_boundary=n_boundary, debug=True)


def construct_source(n_size):
    """ Construct a sparse-matrix source with some points at the corners and in the center"""
    locations = torch.tensor([
        [n_size[0] // 2, 0, 0, 0],
        [n_size[1] // 2, 0, 0, 0],
        [n_size[2] // 2, 0, n_size[2] - 1], 0])

    return torch.sparse_coo_tensor(locations, torch.tensor([1, 1, 1, 1]), n_size, dtype=dtype)


@pytest.mark.parametrize("n_size", [(128, 100, 93), (50, 49, 1)])
def test_propagator(n_size: tuple[int, int, int]):
    """Tests the forward and inverse propagator

    The wavesim algorithm only needs the propagator (L+1)^(-1) to be implemented.
    For testing, and for evaluating the final residue, the Domain and MultiDomain classes
    also implement the 'inverse propagator L+1', which is basically the homogeneous part of the forward operator A.

    This test checks that the forward and inverse propagator are consistent, namely (L+1)^(-1) (L+1) x = x.
    todo: check if the operators are actually correct (not just consistent)
    Note that the propagator is domain-local, so the wrapping correction and domain
    transfer functions are not tested here.
    """

    # construct the (multi-) domain operator
    domain = construct_domain(n_size, n_boundary=8)

    # generate a random vector x
    x = random_vector(n_size)[..., None]
    x = torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)  # add polarization dimension

    # assert that (L+1) (L+1)^-1 x = x
    domain.set(0, x)
    domain.propagator(0, 0)
    domain.inverse_propagator(0, 0)
    x_reconstructed = domain.get(0)
    assert allclose(x, x_reconstructed)

    # also assert that (L+1)^-1 (L+1) x = x, use different slots for input and output
    domain.set(0, x)
    domain.inverse_propagator(0, 1)
    domain.propagator(1, 1)
    x_reconstructed = domain.get(1)
    assert allclose(x, x_reconstructed)

    # for the non-decomposed case, test if the propagator gives the correct value
    n_size = torch.tensor(n_size, dtype=torch.float64)
    # choose |k| <  Nyquist, make sure k is at exact grid point in Fourier space
    k_relative = torch.tensor((0.2, -0.15, 0.4), dtype=torch.float64)
    k = 2 * torch.pi * torch.round(k_relative * n_size) / n_size  # in 1/pixels
    k[n_size == 1] = 0.0
    plane_wave = torch.exp(1j * (
        k[0] * torch.arange(n_size[0], device=domain.device).reshape(-1, 1, 1) +
        k[1] * torch.arange(n_size[1], device=domain.device).reshape(1, -1, 1) +
        k[2] * torch.arange(n_size[2], device=domain.device).reshape(1, 1, -1)))[..., None]
    domain.set(0, plane_wave)
    domain.inverse_propagator(0, 0)
    result = domain.get(0)
    laplace_kernel = (k[0]**2 + k[1]**2 + k[2]**2) / domain.pixel_size ** 2  # -∇² [negative of laplace kernel]
    correct_result = (1.0 + domain.scale * (laplace_kernel + domain.shift)) * plane_wave  # L+1 =  scale·(-∇²) + 1.
    # note: the result is not exactly the same because wavesim is using the real-space kernel, and we compare to
    # the Fourier-space kernel
    assert allclose(result, correct_result, rtol=0.01)
