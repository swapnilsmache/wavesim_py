import pytest
import numpy as np
from helmholtzbase import HelmholtzBase
from save_details import relative_error


@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_accretive(wrap_correction):
    """ Check that sum of propagator and medium operators (L + V) is accretive
        , i.e., has a non-negative real part"""
    n = np.ones((256, 1, 1)).astype(np.float32)
    source = np.zeros_like(n)
    source[0] = 1.
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction=wrap_correction)
    l_plus_1_inv = base.propagator(np.eye(base.domain_size[0], dtype=np.float32), base.scaling[base.domains_iterator[0]])[:base.domain_size[0],:base.domain_size[0]]
    l_plus_1 = np.linalg.inv(l_plus_1_inv)
    b = base.medium_operators[(0, 0, 0)](np.eye(base.domain_size[0], dtype=np.float32))
    a = l_plus_1 - b

    acc = np.min(np.real(np.linalg.eigvals(a + np.asarray(np.conj(a).T))))
    print(f'{acc:.2e}')
    assert np.round(acc, 5) >= 0, f'a is not accretive. {acc}'


@pytest.mark.parametrize("wrap_correction", [None, 'wrap_corr', 'L_omega'])
def test_contraction(wrap_correction):
    """ Check that potential V is a contraction,
        i.e., the operator norm ||V|| < 1 """
    n = np.ones((256, 1, 1)).astype(np.float32)
    source = np.zeros_like(n)
    source[0] = 1.
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction=wrap_correction)
    vc = np.max(np.abs(base.v))
    assert vc < 1, f'||V|| not < 1, but {vc}'


def test_subdomain_op_reconstruction():
    """ Check splitting of operators into subdomains still gives the original operator after reconstruction """
    n = np.ones((256, 1, 1)).astype(np.float32)
    source = np.zeros_like(n)
    source[0] = 1.

    # Get the operator A = (L+1)-B = (L+1)-(1-V) = L+V for the full-domain problem (baseline to compare against)
    base = HelmholtzBase(n=n, source=source, n_domains=1, wrap_correction='wrap_corr', cp=296)
    x = np.eye(base.domain_size[0], dtype=np.float32)
    l_plus_1_inv = base.propagator(x, base.scaling[base.domains_iterator[0]])
    l_plus_1 = np.linalg.inv(l_plus_1_inv)
    b = base.medium_operators[base.domains_iterator[0]](x)
    a = l_plus_1 - b

    # Get the subdomain operators and transfer corrections (2 subdomains) and reconstruct A
    base2 = HelmholtzBase(n=n, source=source, n_domains=2, wrap_correction='wrap_corr', cp=296)
    sub_n = base2.domain_size[0]
    x_ = np.eye(sub_n, dtype=np.float32)

    # (L+1) and B for subdomain 1
    l_plus_1_inv_1 = base2.propagator(x_, base2.scaling[base2.domains_iterator[0]])
    l_plus_1_1 = np.linalg.inv(l_plus_1_inv_1)
    b1 = base2.medium_operators[base2.domains_iterator[0]](x_)

    # (L+1) and B for subdomain 2
    l_plus_1_inv_2 = base2.propagator(x_, base2.scaling[base2.domains_iterator[1]])
    l_plus_1_2 = np.linalg.inv(l_plus_1_inv_2)
    b2 = base2.medium_operators[base2.domains_iterator[1]](x_)

    # Transfer corrections
    b1_corr = base2.transfer(x_, base2.scaling[base2.domains_iterator[0]], +1)
    b2_corr = base2.transfer(x_, base2.scaling[base2.domains_iterator[1]], -1)

    # Reconstruct A using above subdomain operators and transfer corrections
    a_reconstructed = np.zeros_like(a, dtype=np.complex64)
    a_reconstructed[:sub_n, :sub_n] = l_plus_1_1 - b1  # Subdomain 1
    a_reconstructed[sub_n:, sub_n:] = l_plus_1_2 - b2  # Subdomain 2
    a_reconstructed[:sub_n, sub_n:] = b1_corr  # Transfer correction
    a_reconstructed[sub_n:, :sub_n] = b2_corr  # Transfer correction
    rel_err = relative_error(a_reconstructed, a)
    print(f'{rel_err:.2e}')
    assert rel_err < 1.e-6, f'operator A not reconstructed properly. relative error high {rel_err:.2e}'
