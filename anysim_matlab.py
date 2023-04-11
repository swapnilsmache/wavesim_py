import numpy as np
from scipy.linalg import eigvals

## AnySim (original implementation in MATLAB as on github: https://github.com/IvoVellekoop/anysim)
def AnySim_matlab(boundaries_width, n, N_roi, pixel_size, k0, s, iters=int(1.e+4)):
    N_dim = n.ndim
    N = int(N_roi+2*boundaries_width)
    bw_l = int(np.floor(boundaries_width))
    bw_r = int(np.ceil(boundaries_width))

    ## make medium
    Vraw = -1j * k0**2 * n**2
    mu_min = 10.0/(boundaries_width * pixel_size)
    mu_min = max(mu_min, 1.e-3 / (N*pixel_size))  # give tiny non-zero minimum value to prevent division by zero in homogeneous media
    Vmin = np.array([np.imag((k0 + 1j * mu_min)**2)])  # minimum required absorption coefficient
    Vmax = 0.95
    ## find scaling factors (and apply scaling and shift to Vraw)
    Tl, Tr, V0, V = center_scale(Vraw, Vmin, Vmax)
    Tl = Tl * 1j

    ## Check that ||V|| < 1 (0.95 here)
    vc = checkV(V)
    if vc < 1:
        pass
    else:
        return print('||V|| not < 1, but {}'.format(vc))

    ## B = 1 - V, and pad
    B = pad_func((1-V), boundaries_width, bw_l, bw_r, N_roi, N_dim, 0).astype('complex_')  # obj.medium(), i.e., medium (1-V)
    medium = lambda x: B * x

    ## make propagator
    L = (coordinates_f(1, N, pixel_size)**2).T
    for d in range(2,N_dim+1):
        L = L + coordinates_f(d, N, pixel_size)**2
    L = Tl * Tr * (L - 1j*V0)
    Lr = np.squeeze(1/(L+1))
    propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn(x)))

    ## Check that A = L + V is accretive
    if N_dim == 1:
        L = np.diag(np.squeeze(L)[bw_l:-bw_r])
        V = np.diag(V)
    else:
        L = L[bw_l:-bw_r, bw_l:-bw_r]
    A = L + V
    for _ in range(10): ## Repeat test for multiple random vectors
        z = np.random.rand(s.shape[0],1) + 1j*np.random.randn(s.shape[0],1)
        acc = np.real(np.matrix(z).H @ A @ z)
        if np.round(acc, 12) < 0:
            return print('A is not accretive. ', acc)

    ## pad the source term with zeros so that it is the same size as B.shape[0]
    b = np.pad(s, N_dim*((bw_l,bw_r),), mode='constant') * np.asarray(Tl.flatten()) # source term y

    ## update
    u = (np.zeros_like(b, dtype='complex_'))    # field u, initialize with 0
    alpha = 0.75                                # ~step size of the Richardson iteration \in (0,1]
    for _ in range(iters):
        t1 = medium(u) + b
        t1 = propagator(t1)
        t1 = medium(u-t1)
        u = u - (alpha * t1)
    u = (Tr.flatten()*u).astype('complex_')

    if N_dim == 1:
        u = u[bw_l:-bw_r]
    elif N_dim == 2:
        u = u[bw_l:-bw_r,bw_l:-bw_r]
    return u

## make medium
def center_scale(Vraw, Vrmin, Vmax):
    dim = sum(Vrmin.shape[a-1]>1 for a in range(Vrmin.ndim))
    Vreshape = np.expand_dims(Vraw, axis=tuple(range(2-dim)))
    Vrmin = np.expand_dims(Vrmin, axis=tuple(range(2-Vrmin.ndim)))
    N = Vreshape.shape[0]
    M = Vreshape.shape[1]
    centers = np.zeros((N, M), dtype='complex_')
    radii = np.zeros((N, M))

    for n in range(N):
        for m in range(M):
            c, r = smallest_circle((Vreshape[n,m,:]))
            # adjust centers and radii so that the real part of centers+radii >= Vmin
            re_diff = np.real(c) + r - Vrmin[n, m]
            if re_diff < 0:
                c = c - re_diff/2
                r = r - re_diff/2
                print('slowing down simulation to accomodate boundary conditions')
            centers[n,m] = c
            radii[n,m] = r
    
    if dim == 0: # potential is a scalar field
        if Vraw.any() < 0:
            return print('Vraw is not accretive')
        else:
            Ttot = Vmax/radii
            Tl = np.sqrt(Ttot)
            Tr  = Tl.copy()
            V0 = centers.flatten()
            V = Ttot.flatten() * ( Vraw - V0 )
    elif dim == 1: # potential is a field of diagonal matrices, stored as column vectors
        if Vraw.any() < 0:
            return print('Vraw is not accretive')
        else:
            if (radii < np.abs(centers) * 1.e-6).any():
                radii = np.maximum(radii, np.abs(c)*1.e-6)
                print('At least one of the components of the potential is (near-)constant, using threshold to avoid divergence in Tr')
            TT = Vmax/radii.flatten()
            Tl = np.sqrt(np.diag(TT))
            Tr = Tl
            V0 = np.diag(centers.flatten())
            V = TT * (Vraw - np.diag(V0))
    # elif dim == 2: # potential is a field of full matrices, stored as pages
    #     # check if matrix is near-singular
    #     # and add a small offset to the singular values if it is
    #     U, S, Vh = np.linalg.svd(radii)
    #     V = Vh.T
    #     cS = np.diag(U.T @ centers @ V)
    #     if (np.diag(S) < np.abs(cS) * 1.e-6).any():
    #         S = np.maximum(S, np.abs(cS)*1.e-6)
    #         radii = U @ S @ V
    #         ('At least one of the components of the potential is (near-)constant, using threshold to avoid divergence in Tr')
    #     # compute scaling factor for the matrix
    #     P, R, C = equilibriate(radii) ### python equivalent function? scipy.linalg.matrix_balance?
    return Tl.flatten(), Tr.flatten(), V0, V

def smallest_circle(points, tolerance=1.e-10):
    points = points.flatten()
    if np.isreal(points).all():
        pmin = np.min(points)
        pmax = np.max(points)
        center = (pmin + pmax)/2
        radius = pmax - center
        return center, radius

    N_reads = 0

    # Step 0, pick four initial corner points based on bounding box
    corner_i = np.zeros(4, dtype=int)
    corner_i[0] = np.argmin(np.real(points))
    corner_i[1] = np.argmax(np.real(points))
    corner_i[2] = np.argmin(np.imag(points))
    corner_i[3] = np.argmax(np.imag(points))
    p_o = np.zeros_like(corner_i, dtype='complex')
    for a in range(corner_i.size):
        p_o[a] = points[corner_i[a]]
    width = np.real(p_o[1] - p_o[0])
    height = np.imag(p_o[3] - p_o[2])
    r_o = (np.sqrt(width**2 + height**2) / 2).astype('complex_')
    center = 0.5 * (np.real(p_o[0] + p_o[1]) + 1j * np.imag(p_o[2] + p_o[3]))
    N_reads = N_reads + points.size

    for _ in range(50):
        # step 1
        '''
        here, p_o contains up to 7 points, pick the 2-3 that correspond
        to the smallest circle enclosing them all
        sort in order of increasing distance from center since it is
        more likely that the final circle will be built from the points
        further away.
        '''
        ind = np.argsort(np.abs(p_o - center))
        center, radius, p_o = smallest_circle_brute_force(np.array([p_o[a] for a in ind]))

        # step 2
        c_c = conjugate_inflated_triangle(p_o, r_o)

        # 2a: select points
        try:
            distances = np.abs(np.expand_dims(points, axis=1) - np.expand_dims(np.concatenate(([center], c_c)), axis=0))
        except:
            distances = np.abs(np.expand_dims(points, axis=1) - np.expand_dims(np.concatenate((center, c_c)), axis=0))

        # print('distances', distances.shape, distances)
        keep = np.max(distances, 1) > r_o# - tolerance
        N_reads = N_reads + keep.size
        
        # 2b: determine outliers
        r_out = np.max(distances, 0)
        outliers_i = np.argmax(distances, 0)
        if r_out.flatten()[0] < radius + tolerance:
            radius = r_out.flatten()[0]
            return center, radius
        outliers = points[outliers_i]
        r_o = np.minimum(np.min(r_out), r_o)
        points = np.concatenate((points[keep], p_o))
        p_o = np.concatenate((outliers, p_o))

    return center, radius

def smallest_circle_brute_force(points, tolerance=1.e-10):

    N = len(points)
    if N==1:
        center = points.copy()
        radius = 0
        corners = points.copy()
        return center, radius, corners
    elif N==2:
        center = (points[0] + points[1])/2
        radius = np.abs(points[0]-center)
        corners = points.copy()
        return center, radius, corners
        
    '''
    Remove one point and recursively construct a smallest circle for the
    remaining points. If the removed point is inside that circle, return
    if the removed point is not in the circle, repeat with a different point
    omitted. First check if it is possible to construct a circle from just 2 points,
    including the third
    todo: faster check to see if two or three points are needed?
    '''
    Ns = np.arange(N)
    for p in range(N):
        reduced = points[[Ns[a]!=p for a in range(len(Ns))]].copy()
        center, radius, corners = smallest_circle_brute_force(reduced, tolerance=1.e-10)
        if np.abs(points[p]-center) <= radius + tolerance:
            return center, radius, corners
        
    # if we get here, no suitable subset was found. This is only possible for 3 points

    ## All three points are edge points
    # now write in matrix form and let Python (MATLAB) solve it
    A = points[0]
    B = points[1]
    C = points[2]
    M = 2 * np.array([[np.real(A)-np.real(B), np.imag(A)-np.imag(B)],[np.real(A)-np.real(C), np.imag(A)-np.imag(C)]])
    b = np.array([[np.abs(A)**2 - np.abs(B)**2],[np.abs(A)**2 - np.abs(C)**2]])

    try:
        c = np.linalg.solve(M, b)
    except:
        c= np.linalg.lstsq(M, b)
    center = c[0] + 1j * c[1]
    radius = np.abs(A-center)
    corners = points

    return center, radius, corners

def conjugate_inflated_triangle(points, r):
    c_c = np.zeros(3, dtype='complex_')
    Np = len(points)
    if Np==2:
        B = points[0]
        C = points[1]
        M_start = (C+B)/2
        M_dir = 1j * (C-B)/np.abs(C-B)
        w = np.abs(C - M_start)
        alpha = np.sqrt(r**2 - w**2)
        c_c[0] = M_start - alpha*M_dir
        c_c[1] = M_start + alpha*M_dir

        # not needed, but we can pick one point 'for free'
        M_dir = (C-B) / np.abs(C-B)
        c_c[2] = B + M_dir*r
        return c_c
    
    ss = signed_surface(points)
    if ss < 0: # reverse direction of circle (otherwise we get the solutions outside of the circle)
        tmp = points[0]
        points[0] = points[2]
        points[2] = tmp

    for p in range(Np):
        '''
        For conjugating point A, define mid-point M_CB and line M_CB-A
        c_A is on this line, at a distance of r from point B (and point
        C)
        %
        c_A = M_CB + alpha * (A - M_CB) / |A-M_CB|  with alpha >= 0 
        w = |C - M_CB|
        h = alpha
        w^2 + alpha^2 = r^2
        alpha = sqrt(r^2 - w^2)
        '''
        B = points[p]
        C = points[np.remainder(p+1, Np)]
        M_start = (C+B)/2
        M_dir = 1j * (C-B)/np.abs(C-B)
        w = np.abs(C - M_start)
        alpha = np.sqrt(r**2 - w**2)
        c_c[p] = M_start + alpha*M_dir
    return c_c

def signed_surface(points):
    return np.imag((points[0]-points[1]) * np.conj(points[2]-points[1]))

## pad (1-V) to size N = N_roi + (2*boundaries_width)
def pad_func(M, boundaries_width, bw_l, bw_r, N_roi, N_dim, element_dimension=0):
    sz = M.shape[element_dimension+0:N_dim]
    if (boundaries_width != 0) & (sz == (1,)):
        M = np.tile(M, (int((np.ones(1) * (N_roi-1) + 1)[0]),))
    M = np.pad(M, ((bw_l,bw_r)), mode='edge')

    for d in range(N_dim):
        try:
            w = boundaries_width[d]
        except:
            w = boundaries_width

    if w>0:
        left_boundary = boundaries_window(np.floor(w))
        right_boundary = boundaries_window(np.ceil(w))
        full_filter = np.concatenate((left_boundary, np.ones((N_roi,1)), np.flip(right_boundary)))
        if N_dim == 1:
            M = M * np.squeeze(full_filter)
        else:
            M = full_filter.T * M * full_filter
    return M

def boundaries_window(L):
    x = np.expand_dims(np.arange(L)/(L-1), axis=1)
    a2 = np.expand_dims(np.array([-0.4891775, 0.1365995/2, -0.0106411/3]) / (0.3635819 * 2 * np.pi), axis=1)
    return np.sin(x * np.expand_dims(np.array([1, 2, 3]), axis=0) * 2 * np.pi) @ a2 + x

## DFT matrix
def DFT_matrix(N):
    l, m = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    return np.power( omega, l * m )

def coordinates_f(dimension, N, pixel_size):
    pixel_size_f = 2 * np.pi/(pixel_size*N)
    return np.expand_dims( fft_range(N) * pixel_size_f, axis=tuple(range(2-dimension)))

def fft_range(N):
    return np.fft.ifftshift(symrange(N))

def symrange(N):
    return range(-int(np.floor(N/2)),int(np.ceil(N/2)))

## Relative error
def relative_error(E_, E_true):
    return np.mean( np.linalg.norm(E_-E_true, ord=2) ) / np.mean( np.linalg.norm(E_true, ord=2) )

## Check that V is a contraction
def checkV(A):
    return np.max(np.abs(A))