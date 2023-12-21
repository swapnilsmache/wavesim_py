import pytest
import numpy as np
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse import diags as spdiags

from helmholtzbase import HelmholtzBase
from utilities import full_matrix, relative_error


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((15, 16)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_accretive(n, boundary_widths, wrap_correction):
    """ Check that operator A = L + V (sum of propagator and medium operators) is accretive
        , i.e., has a non-negative real part """
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    patch = (0, 0, 0)
    if wrap_correction == 'L_omega':
        l_plus_1_operator = lambda x: (np.fft.ifftn((base.scaling[patch] * base.l_p + 1) *
                                       np.fft.fftn(x, base.domain_size*base.omega)))[
                                       tuple([slice(0, base.domain_size[i]) for i in range(3)])]
    else:
        l_plus_1_operator = lambda x: np.fft.ifftn((base.scaling[patch] * base.l_p + 1) * np.fft.fftn(x))
    l_plus_1 = full_matrix(l_plus_1_operator, base.domain_size)
    b = full_matrix(base.medium_operators[patch], base.domain_size)
    a = (l_plus_1 - b).todense()

    acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))
    print(f'acc {acc:.2e}')
    assert np.round(acc, 5) >= 0, f'a is not accretive. {acc}'


@pytest.mark.parametrize("n, boundary_widths", [(np.ones(256), 0), (np.ones(256), 20), 
                                                (np.ones((20, 21)), 0), (np.ones((15, 16)), 5),
                                                (np.ones((5, 6, 7)), 0), (np.ones((5, 6, 7)), 1)])
@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_contraction(n, boundary_widths, wrap_correction):
    """ Check that potential V is a contraction,
        i.e., the operator norm ||V|| < 1 """
    n_correction = 8
    if n.ndim == 3:
        n_correction = 2
    base = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                         wrap_correction=wrap_correction, n_correction=n_correction)
    patch = (0, 0, 0)
    if wrap_correction == 'wrap_corr':
        scaling = base.scaling[patch]
        v_corr = spdiags(base.v.ravel(), dtype=np.complex64) - scaling * full_matrix(base.wrap_corr, base.domain_size)
    else:
        # vc = np.max(np.abs(base.v))
        v_corr = (full_matrix(base.medium_operators[patch], base.domain_size) 
                  - spdiags(np.ones(np.prod(base.domain_size)), dtype=np.complex64))
    vc = spnorm(v_corr, 2)
    print(f'vc {vc:.2f}')
    assert vc < 1, f'||V|| not < 1, but {vc}'


@pytest.mark.parametrize("n, boundary_widths, n_correction", [(np.ones(256), 0, 64), (np.ones(256), 10, 64), 
                                                              (np.ones((20, 21)), 0, 7), (np.ones((15, 16)), 5, 7),
                                                              (np.ones((5, 6, 7)), 0, 2), (np.ones((5, 6, 7)), 1, 2)])
def test_compare_A(n, boundary_widths, n_correction):
    """ Check that the operator (A) is the same for wrap_correction = ['wrap_corr', 'L_omega'] """
    base_w = HelmholtzBase(n=n, boundary_widths=boundary_widths, 
                           wrap_correction='wrap_corr', n_correction=n_correction)
    patch = (0, 0, 0)
    scaling_w = base_w.scaling[patch]
    l_w_operator = lambda x: np.fft.ifftn((scaling_w * base_w.l_p + 1) * np.fft.fftn(x))
    l_w_plus1 = full_matrix(l_w_operator, base_w.domain_size)
    # v_w = spdiags(base_w.v.ravel(), dtype=np.complex64) - scaling_w*full_matrix(base_w.wrap_corr, base_w.domain_size)
    # a_w = (l_w_plus1 + v_w)/scaling_w
    b_w = full_matrix(base_w.medium_operators[patch], base_w.domain_size)
    a_w = ((l_w_plus1 - b_w)/scaling_w).todense()

    base_o = HelmholtzBase(n=n, boundary_widths=boundary_widths, wrap_correction='L_omega')    
    scaling_o = base_o.scaling[patch]
    l_o_operator = lambda x: (np.fft.ifftn((scaling_o * base_o.l_p + 1) *
                                           np.fft.fftn(x, base_o.domain_size*base_o.omega)))[
                                           tuple([slice(0, base_o.domain_size[i]) for i in range(3)])]
    l_o_plus1 = full_matrix(l_o_operator, base_w.domain_size)
    # v_o = np.diag(base_o.v.ravel())
    # a_o = (l_o_plus1 + v_o)/scaling_o
    b_o = full_matrix(base_o.medium_operators[patch], base_w.domain_size)
    a_o = ((l_o_plus1 - b_o)/scaling_o).todense()

    # base = HelmholtzBase(n=n, boundary_widths=boundary_widths, wrap_correction=None)
    # l_plus_1_operator = lambda x: np.fft.ifftn((base.scaling[patch] * base.l_p + 1) * np.fft.fftn(x))
    # l_plus_1 = full_matrix(l_plus_1_operator, base_w.domain_size)
    # b = full_matrix(base.medium_operators[patch], base_w.domain_size)
    # a = (l_plus_1 - b)/base.scaling[patch]

    if boundary_widths == 0:
        rel_err = relative_error(a_w, a_o)
    else:
        crop2roi = tuple([slice(base_w.boundary_pre[0], -base_w.boundary_post[0]) 
                          for _ in range(2)])  # crop array from n_ext to n_roi

        rel_err = relative_error(a_w[crop2roi], a_o[crop2roi])

    print(f'{rel_err:.2e}')
    # print(f'{rel_err:.2e}, {relative_error(a, a_o):.2e}, {relative_error(a, a_w):.2e}')
    assert rel_err <= 3.e-2, f'Operator A (wrap_corr case) != A (L_omega case). relative error {rel_err:.2e}'


# def test_subdomain_op_reconstruction():
#     """ Check splitting of operators into subdomains still gives the original operator after reconstruction """
#     n = np.ones((256, 1, 1), dtype=np.float32)
#     source = np.zeros_like(n)
#     source[0] = 1.

#     # Get the operator A = (L+1)-B = (L+1)-(1-V) = L+V for the full-domain problem (baseline to compare against)
#     base = HelmholtzBase(n=n, n_domains=1, wrap_correction='wrap_corr')
#     x = np.eye(base.domain_size[0], dtype=np.float32)
#     l_plus_1_inv = base.propagator(x, base.scaling[base.domains_iterator[0]])
#     l_plus_1 = np.linalg.inv(l_plus_1_inv)
#     b = base.medium_operators[base.domains_iterator[0]](x)
#     a = l_plus_1 - b

#     # Get the subdomain operators and transfer corrections (2 subdomains) and reconstruct A
#     base2 = HelmholtzBase(n=n, n_domains=2, wrap_correction='wrap_corr')
#     sub_n = base2.domain_size[0]
#     x_ = np.eye(sub_n, dtype=np.float32)

#     # (L+1) and B for subdomain 1
#     l_plus_1_inv_1 = base2.propagator(x_, base2.scaling[base2.domains_iterator[0]])
#     l_plus_1_1 = np.linalg.inv(l_plus_1_inv_1)
#     b1 = base2.medium_operators[base2.domains_iterator[0]](x_)

#     # (L+1) and B for subdomain 2
#     l_plus_1_inv_2 = base2.propagator(x_, base2.scaling[base2.domains_iterator[1]])
#     l_plus_1_2 = np.linalg.inv(l_plus_1_inv_2)
#     b2 = base2.medium_operators[base2.domains_iterator[1]](x_)

#     # Transfer corrections
#     b1_corr = base2.transfer(x_, base2.scaling[base2.domains_iterator[0]], +1)
#     b2_corr = base2.transfer(x_, base2.scaling[base2.domains_iterator[1]], -1)

#     # Reconstruct A using above subdomain operators and transfer corrections
#     a_reconstructed = np.zeros_like(a, dtype=np.complex64)
#     a_reconstructed[:sub_n, :sub_n] = l_plus_1_1 - b1  # Subdomain 1
#     a_reconstructed[sub_n:, sub_n:] = l_plus_1_2 - b2  # Subdomain 2
#     a_reconstructed[:sub_n, sub_n:] = b1_corr  # Transfer correction
#     a_reconstructed[sub_n:, :sub_n] = b2_corr  # Transfer correction
#     rel_err = relative_error(a_reconstructed, a)
#     print(f'Relative error between A reconstructed and A: {rel_err:.2e}')
#     assert rel_err <= 1.e-6, f'operator A not reconstructed properly. relative error high {rel_err:.2e}'
