import datetime
import warnings

print("time 2.1: " + datetime.datetime.now().isoformat())
print("time 2.2: " + datetime.datetime.now().isoformat())
print("time 2.3: " + datetime.datetime.now().isoformat())
import scipy.sparse
print("time 2.4: " + datetime.datetime.now().isoformat())
from numba import njit, prange
print("time 2.5: " + datetime.datetime.now().isoformat())
import numpy as np
print("time 2.6: " + datetime.datetime.now().isoformat())


def stack_images(*images):
    """This function stacks images along the third axis.
    This is useful for combining e.g. rgb color channels or color and alpha channels.

    Parameters
    ----------
    *images: numpy.ndarray
        Images to be stacked.

    Returns
    -------
    image: numpy.ndarray
        Stacked images as numpy.ndarray

    Example
    -------
    >>> from pymatting.util.util import stack_images
    >>> import numpy as np
    >>> I = stack_images(np.random.rand(4,5,3), np.random.rand(4,5,3))
    >>> I.shape
    (4, 5, 6)
    """
    images = [
        (image if len(image.shape) == 3 else image[:, :, np.newaxis])
        for image in images
    ]
    return np.concatenate(images, axis=2)

print("time 2.7: " + datetime.datetime.now().isoformat())


@njit("void(f4[:, :, :], f4[:, :, :])", cache=True, nogil=True, parallel=True)
def _resize_nearest_multichannel(dst, src):
    """
    Internal method.

    Resize image src to dst using nearest neighbors filtering.
    Images must have multiple color channels, i.e. :code:`len(shape) == 3`.

    Parameters
    ----------
    dst: numpy.ndarray of type np.float32
        output image
    src: numpy.ndarray of type np.float32
        input image
    """
    h_src, w_src, depth = src.shape
    h_dst, w_dst, depth = dst.shape

    for y_dst in prange(h_dst):
        for x_dst in range(w_dst):
            x_src = max(0, min(w_src - 1, x_dst * w_src // w_dst))
            y_src = max(0, min(h_src - 1, y_dst * h_src // h_dst))

            for c in range(depth):
                dst[y_dst, x_dst, c] = src[y_src, x_src, c]


print("time 2.8: " + datetime.datetime.now().isoformat())


@njit("void(f4[:, :], f4[:, :])", cache=True, nogil=True, parallel=True)
def _resize_nearest(dst, src):
    """
    Internal method.

    Resize image src to dst using nearest neighbors filtering.
    Images must be grayscale, i.e. :code:`len(shape) == 3`.

    Parameters
    ----------
    dst: numpy.ndarray of type np.float32
        output image
    src: numpy.ndarray of type np.float32
        input image
    """
    h_src, w_src = src.shape
    h_dst, w_dst = dst.shape

    for y_dst in prange(h_dst):
        for x_dst in range(w_dst):
            x_src = max(0, min(w_src - 1, x_dst * w_src // w_dst))
            y_src = max(0, min(h_src - 1, y_dst * h_src // h_dst))

            dst[y_dst, x_dst] = src[y_src, x_src]


print("time 2.9: " + datetime.datetime.now().isoformat())

# TODO
# There should be an option to switch @njit(parallel=True) on or off.
# parallel=True would be faster, but might cause race conditions.
# User should have the option to turn it on or off.
@njit("Tuple((f4[:, :, :], f4[:, :, :]))(f4[:, :, :], f4[:, :], f4, i4, i4, i4, f4)", cache=True, nogil=True)
def _estimate_fb_ml(
    input_image,
    input_alpha,
    regularization,
    n_small_iterations,
    n_big_iterations,
    small_size,
    gradient_weight,
):
    h0, w0, depth = input_image.shape

    dtype = np.float32

    w_prev = 1
    h_prev = 1

    F_prev = np.empty((h_prev, w_prev, depth), dtype=dtype)
    B_prev = np.empty((h_prev, w_prev, depth), dtype=dtype)

    n_levels = int(np.ceil(np.log2(max(w0, h0))))

    for i_level in range(n_levels + 1):
        w = round(w0 ** (i_level / n_levels))
        h = round(h0 ** (i_level / n_levels))

        image = np.empty((h, w, depth), dtype=dtype)
        alpha = np.empty((h, w), dtype=dtype)

        _resize_nearest_multichannel(image, input_image)
        _resize_nearest(alpha, input_alpha)

        F = np.empty((h, w, depth), dtype=dtype)
        B = np.empty((h, w, depth), dtype=dtype)

        _resize_nearest_multichannel(F, F_prev)
        _resize_nearest_multichannel(B, B_prev)

        if w <= small_size and h <= small_size:
            n_iter = n_small_iterations
        else:
            n_iter = n_big_iterations

        b = np.zeros((2, depth), dtype=dtype)

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i_iter in range(n_iter):
            for y in prange(h):
                for x in range(w):
                    a0 = alpha[y, x]
                    a1 = 1.0 - a0

                    a00 = a0 * a0
                    a01 = a0 * a1
                    # a10 = a01 can be omitted due to symmetry of matrix
                    a11 = a1 * a1

                    for c in range(depth):
                        b[0, c] = a0 * image[y, x, c]
                        b[1, c] = a1 * image[y, x, c]

                    for d in range(4):
                        x2 = max(0, min(w - 1, x + dx[d]))
                        y2 = max(0, min(h - 1, y + dy[d]))

                        gradient = abs(a0 - alpha[y2, x2])

                        da = regularization + gradient_weight * gradient

                        a00 += da
                        a11 += da

                        for c in range(depth):
                            b[0, c] += da * F[y2, x2, c]
                            b[1, c] += da * B[y2, x2, c]

                    determinant = a00 * a11 - a01 * a01

                    inv_det = 1.0 / determinant

                    b00 = inv_det * a11
                    b01 = inv_det * -a01
                    b11 = inv_det * a00

                    for c in range(depth):
                        F_c = b00 * b[0, c] + b01 * b[1, c]
                        B_c = b01 * b[0, c] + b11 * b[1, c]

                        F_c = max(0.0, min(1.0, F_c))
                        B_c = max(0.0, min(1.0, B_c))

                        F[y, x, c] = F_c
                        B[y, x, c] = B_c

        F_prev = F
        B_prev = B

        w_prev = w
        h_prev = h

    return F, B


print("time 2.10: " + datetime.datetime.now().isoformat())


def estimate_foreground_ml(
    image,
    alpha,
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    return_background=False,
    gradient_weight=1.0,
):
    """Estimates the foreground of an image given its alpha matte.

    See :cite:`germer2020multilevel` for reference.

    Parameters
    ----------
    image: numpy.ndarray
        Input image with shape :math:`h \\times  w \\times d`
    alpha: numpy.ndarray
        Input alpha matte shape :math:`h \\times  w`
    regularization: float
        Regularization strength :math:`\\epsilon`, defaults to :math:`10^{-5}`.
        Higher regularization results in smoother colors.
    n_small_iterations: int
        Number of iterations performed on small scale, defaults to :math:`10`
    n_big_iterations: int
        Number of iterations performed on large scale, defaults to :math:`2`
    small_size: int
        Threshold that determines at which size `n_small_iterations` should be used
    return_background: bool
        Whether to return the estimated background in addition to the foreground
    gradient_weight: float
        Larger values enforce smoother foregrounds, defaults to :math:`1`

    Returns
    -------
    F: numpy.ndarray
        Extracted foreground
    B: numpy.ndarray
        Extracted background

    Example
    -------
     from pymatting import *
     image = load_image("data/lemur/lemur.png", "RGB")
     alpha = load_image("data/lemur/lemur_alpha.png", "GRAY")
     F = estimate_foreground_ml(image, alpha, return_background=False)
     F, B = estimate_foreground_ml(image, alpha, return_background=True)

    See Also
    ----
    stack_images: This function can be used to place the foreground on a new background.
    """

    foreground, background = _estimate_fb_ml(
        image.astype(np.float32),
        alpha.astype(np.float32),
        regularization,
        n_small_iterations,
        n_big_iterations,
        small_size,
        gradient_weight,
    )

    if return_background:
        return foreground, background

    return foreground


print("time 2.11: " + datetime.datetime.now().isoformat())


def cg(
    A,
    b,
    x0=None,
    atol=0.0,
    rtol=1e-7,
    maxiter=10000,
    callback=None,
    M=None,
    reorthogonalize=False,
):
    """Solves a system of linear equations :math:`Ax=b` using conjugate gradient descent :cite:`hestenes1952methods`

    Parameters
    ----------
    A: scipy.sparse.csr_matrix
       Square matrix
    b: numpy.ndarray
       Vector describing the right-hand side of the system
    x0: numpy.ndarray
       Initialization, if `None` then :code:`x=np.zeros_like(b)`
    atol: float
       Absolute tolerance. The loop terminates if the :math:`||r||` is smaller than `atol`, where :math:`r` denotes the residual of the current iterate.
    rtol: float
       Relative tolerance. The loop terminates if :math:`{||r||}/{||b||}` is smaller than `rtol`, where :math:`r` denotes the residual of the current iterate.
    callback: function
       Function :code:`callback(A, x, b, norm_b, r, norm_r)` called after each iteration, defaults to `None`
    M: function or scipy.sparse.csr_matrix
       Function that applies the preconditioner to a vector. Alternatively, `M` can be a matrix describing the precondioner.
    reorthogonalize: boolean
        Wether to apply reorthogonalization of the residuals after each update, defaults to `False`


    Returns
    -------
    x: numpy.ndarray
        Solution of the system

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> A = np.array([[3.0, 1.0], [1.0, 2.0]])
    >>> M = jacobi(A)
    >>> b = np.array([4.0, 3.0])
    >>> cg(A, b, M=M)
    array([1., 1.])
    """
    if M is None:

        def precondition(x):
            return x

    elif callable(M):
        precondition = M
    else:

        def precondition(x):
            return M.dot(x)

    x = np.zeros_like(b) if x0 is None else x0.copy()

    norm_b = np.linalg.norm(b)

    if callable(A):
        r = b - A(x)
    else:
        r = b - A.dot(x)

    norm_r = np.linalg.norm(r)

    if norm_r < atol or norm_r < rtol * norm_b:
        return x

    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)

    for iteration in range(maxiter):
        r_old = r.copy()

        if callable(A):
            Ap = A(p)
        else:
            Ap = A.dot(p)

        alpha = rz / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        norm_r = np.linalg.norm(r)

        if callback is not None:
            callback(A, x, b, norm_b, r, norm_r)

        if norm_r < atol or norm_r < rtol * norm_b:
            return x

        z = precondition(r)

        if reorthogonalize:
            beta = np.inner(r - r_old, z) / rz
            rz = np.inner(r, z)
        else:
            beta = 1.0 / rz
            rz = np.inner(r, z)
            beta *= rz

        p *= beta
        p += z

    raise ValueError(
        "Conjugate gradient descent did not converge within %d iterations" % maxiter
    )


print("time 2.12: " + datetime.datetime.now().isoformat())


@njit("void(f8[:], i8[:], i8[:], f8[:], i8)", cache=True, nogil=True)
def _backsub_L_csc_inplace(data, indices, indptr, x, n):
    for j in range(n):
        k = indptr[j]
        L_jj = data[k]
        temp = x[j] / L_jj

        x[j] = temp

        for k in range(indptr[j] + 1, indptr[j + 1]):
            i = indices[k]
            L_ij = data[k]

            x[i] -= L_ij * temp


@njit("void(f8[:], i8[:], i8[:], f8[:], i8)", cache=True, nogil=True)
def _backsub_LT_csc_inplace(data, indices, indptr, x, n):
    for i in range(n - 1, -1, -1):
        s = x[i]

        for k in range(indptr[i] + 1, indptr[i + 1]):
            j = indices[k]
            L_ji = data[k]
            s -= L_ji * x[j]

        k = indptr[i]
        L_ii = data[k]

        x[i] = s / L_ii


class CholeskyDecomposition(object):
    """Cholesky Decomposition

    Calling this object applies the preconditioner to a vector by forward and back substitution.

    Parameters
    ----------
    Ltuple: tuple of numpy.ndarrays
        Tuple of array describing values, row indices and row pointers for Cholesky factor in the compressed sparse column format (csc)
    """

    def __init__(self, Ltuple):
        self.Ltuple = Ltuple

    @property
    def L(self):
        """Returns the Cholesky factor

        Returns
        -------
        L: scipy.sparse.csc_matrix
            Cholesky factor
        """
        Lv, Lr, Lp = self.Ltuple
        n = len(Lp) - 1
        return scipy.sparse.csc_matrix(self.Ltuple, (n, n))

    def __call__(self, b):
        Lv, Lr, Lp = self.Ltuple
        n = len(b)
        x = b.copy()
        _backsub_L_csc_inplace(Lv, Lr, Lp, x, n)
        _backsub_LT_csc_inplace(Lv, Lr, Lp, x, n)
        return x


def ichol(
    A,
    discard_threshold=1e-4,
    shifts=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100, 1e3, 1e4, 1e5],
    max_nnz=int(4e9 / 16),
    relative_discard_threshold=0.0,
    diag_keep_discarded=True,
):
    """Implements the thresholded incomplete Cholesky decomposition

    Parameters
    ----------
    A: scipy.sparse.csc_matrix
        Matrix for which the preconditioner should be calculated
    discard_threshold: float
        Values having an absolute value smaller than this threshold will be discarded while calculating the Cholesky decompositions
    shifts: array of floats
        Values to try for regularizing the matrix of interest in case it is not positive definite after discarding the small values
    max_nnz: int
        Maximum number of non-zero entries in the Cholesky decomposition. Defaults to 250 million, which should usually be around 4 GB.
    relative_discard_threshold: float
        Values with an absolute value of less than :code:`relative_discard_threshold * sum(abs(A[j:, j]))` will be discarded.
        A dense ichol implementation with relative threshold would look like this::

            L = np.tril(A)
            for j in range(n):
                col = L[j:, j]
                col -= np.sum(L[j, :j] * L[j:, :j], axis=1)
                discard_mask = abs(col[1:]) < relative_discard_threshold * np.sum(np.abs(A[j:, j]))
                col[1:][discard_mask] = 0
                col[0] **= 0.5
                col[1:] /= col[0]

    diag_keep_discarded: bool
        Whether to update the diagonal with the discarded values. Usually better if :code:`True`.

    Returns
    -------
    chol: CholeskyDecomposition
        Preconditioner or solver object.

    Raises
    ------
    ValueError
        If inappropriate parameter values were passed

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> A = np.array([[2.0, 3.0], [3.0, 5.0]])
    >>> cholesky_decomposition = ichol(csc_matrix(A))
    >>> cholesky_decomposition(np.array([1.0, 2.0]))
    array([-1.,  1.])
    """

    if isinstance(A, scipy.sparse.csr_matrix):
        A = A.T

    if not isinstance(A, scipy.sparse.csc_matrix):
        raise ValueError("Matrix A must be a scipy.sparse.csc_matrix")

    if not A.has_canonical_format:
        A.sum_duplicates()

    m, n = A.shape

    assert m == n

    Lv = np.empty(max_nnz, dtype=np.float64)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int64)  # Row indices of non-zero elements of L
    Lp = np.zeros(
        n + 1, dtype=np.int64
    )  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv

    for shift in shifts:
        nnz = _ichol(
            n,
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            Lv,
            Lr,
            Lp,
            discard_threshold,
            shift,
            max_nnz,
            relative_discard_threshold,
            diag_keep_discarded,
        )

        if nnz >= 0:
            break

        if nnz == -1:
            print("PERFORMANCE WARNING:")
            print(
                "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with parameters:"
            )
            print("    discard_threshold = %e" % discard_threshold)
            print("    shift = %e" % shift)
            print("Try decreasing discard_threshold or start with a larger shift")
            print("")

        if nnz == -2:
            raise ValueError(
                "Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created. Try increasing max_nnz or discard_threshold."
            )

    if nnz < 0:
        raise ValueError(
            "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A and diagonal shifts did not help."
        )

    Lv = Lv[:nnz]
    Lr = Lr[:nnz]

    return CholeskyDecomposition((Lv, Lr, Lp))


@njit(
    "i8(i8, f8[:], i8[:], i8[:], f8[:], i8[:], i8[:], f8, f8, i8, f8, b1)",
    cache=True,
    nogil=True,
)
def _ichol(
    n,
    Av,
    Ar,
    Ap,
    Lv,
    Lr,
    Lp,
    discard_threshold,
    shift,
    max_nnz,
    relative_discard_threshold,
    diag_keep_discarded,
):
    """
    :cite:`jones1995improved` might be slightly interesting for the general idea
    to use linked list to keep track of the sparse matrix values. But instead of
    pointers, we have to use indices here, since this is Python and not C.
    """
    nnz = 0
    c_n = 0
    s = np.zeros(n, np.int64)  # Next non-zero row index i in column j of L
    t = np.zeros(n, np.int64)  # First subdiagonal index i in column j of A
    l = np.zeros(n, np.int64) - 1  # Linked list of non-zero columns in row k of L
    a = np.zeros(n, np.float64)  # Values of column j
    r = np.zeros(n, np.float64)  # r[j] = sum(abs(A[j:, j])) for relative threshold
    b = np.zeros(
        n, np.bool8
    )  # b[i] indicates if the i-th element of column j is non-zero
    c = np.empty(n, np.int64)  # Row indices of non-zero elements in column j
    d = np.full(n, shift, np.float64)  # Diagonal elements of A
    for j in range(n):
        for idx in range(Ap[j], Ap[j + 1]):
            i = Ar[idx]
            if i == j:
                d[j] += Av[idx]
                t[j] = idx + 1
            if i >= j:
                r[j] += abs(Av[idx])
    for j in range(n):  # For each column j
        for idx in range(t[j], Ap[j + 1]):  # For each L_ij
            i = Ar[idx]
            L_ij = Av[idx]
            if L_ij != 0.0 and i > j:
                a[i] += L_ij  # Assign non-zero value to L_ij in sparse column
                if not b[i]:
                    b[i] = True  # Mark it as non-zero
                    c[c_n] = i  # Remember index for later deletion
                    c_n += 1
        k = l[j]  # Find index k of column with non-zero element in row j
        while k != -1:  # For each column of that type
            k0 = s[k]  # Start index of non-zero elements in column k
            k1 = Lp[k + 1]  # End index
            k2 = l[k]  # Remember next column index before it is overwritten
            L_jk = Lv[k0]  # Value of non-zero element at start of column
            k0 += 1  # Advance to next non-zero element in column
            if k0 < k1:  # If there is a next non-zero element
                s[k] = k0  # Advance start index in column k to next non-zero element
                i = Lr[k0]  # Row index of next non-zero element in column k
                l[k] = l[i]  # Remember old list i index in list k
                l[i] = k  # Insert index of non-zero element into list i
                for idx in range(k0, k1):  # For each non-zero L_ik in column k
                    i = Lr[idx]
                    L_ik = Lv[idx]
                    a[i] -= L_ik * L_jk  # Update element L_ij in sparse column
                    if not b[i]:  # Check if sparse column element was zero
                        b[i] = True  # Mark as non-zero in sparse column
                        c[c_n] = i  # Remember index for later deletion
                        c_n += 1
            k = k2  # Advance to next column k
        if d[j] <= 0.0:
            return -1
        if nnz + 1 + c_n > max_nnz:
            return -2
        d[j] = np.sqrt(d[j])  # Update diagonal element L_ii
        Lv[nnz] = d[j]  # Add diagonal element L_ii to L
        Lr[nnz] = j  # Add row index of L_ii to L
        nnz += 1
        s[j] = nnz  # Set first non-zero index of column j
        for i in np.sort(
            c[:c_n]
        ):  # Sort row indices of column j for correct insertion order into L
            L_ij = a[i] / d[j]  # Get non-zero element from sparse column j
            if diag_keep_discarded:
                d[i] -= L_ij * L_ij  # Update diagonal element L_ii
            rel = (
                relative_discard_threshold * r[j]
            )  # Relative discard threshold (before div)
            if (
                abs(L_ij) > discard_threshold and abs(a[i]) > rel
            ):  # If element is sufficiently non-zero
                if not diag_keep_discarded:
                    d[i] -= L_ij * L_ij  # Update diagonal element L_ii
                Lv[nnz] = L_ij  # Add element L_ij to L
                Lr[nnz] = i  # Add row index of L_ij
                nnz += 1
            a[i] = 0.0  # Set element i in column j to zero
            b[i] = False  # Mark element as zero
        c_n = 0  # Discard row indices of non-zero elements in column j.
        Lp[j + 1] = nnz  # Update count of non-zero elements up to column j
        if Lp[j] + 1 < Lp[j + 1]:  # If column j has a non-zero element below diagonal
            i = Lr[Lp[j] + 1]  # Row index of first off-diagonal non-zero element
            l[j] = l[i]  # Remember old list i index in list j
            l[i] = j  # Insert index of non-zero element into list i
    return nnz


print("time 2.13: " + datetime.datetime.now().isoformat())


@njit("void(f8[:, :, :], f8, i8, f8[:, :, :], i8[:], i8[:], b1[:, :])", cache=True, nogil=True)
def _cf_laplacian(image, epsilon, r, values, indices, indptr, is_known):
    h, w, d = image.shape
    assert d == 3
    size = 2 * r + 1
    window_area = size * size

    for yi in range(h):
        for xi in range(w):
            i = xi + yi * w
            k = i * (4 * r + 1) ** 2
            for yj in range(yi - 2 * r, yi + 2 * r + 1):
                for xj in range(xi - 2 * r, xi + 2 * r + 1):
                    j = xj + yj * w

                    if 0 <= xj < w and 0 <= yj < h:
                        indices[k] = j

                    k += 1

            indptr[i + 1] = k

    # Centered and normalized window colors
    c = np.zeros((2 * r + 1, 2 * r + 1, 3))

    # For each pixel of image
    for y in range(r, h - r):
        for x in range(r, w - r):

            if np.all(is_known[y - r : y + r + 1, x - r : x + r + 1]):
                continue

            # For each color channel
            for dc in range(3):
                # Calculate sum of color channel in window
                s = 0.0
                for dy in range(size):
                    for dx in range(size):
                        s += image[y + dy - r, x + dx - r, dc]

                # Calculate centered window color
                for dy in range(2 * r + 1):
                    for dx in range(2 * r + 1):
                        c[dy, dx, dc] = (
                            image[y + dy - r, x + dx - r, dc] - s / window_area
                        )

            # Calculate covariance matrix over color channels with epsilon regularization
            a00 = epsilon
            a01 = 0.0
            a02 = 0.0
            a11 = epsilon
            a12 = 0.0
            a22 = epsilon

            for dy in range(size):
                for dx in range(size):
                    a00 += c[dy, dx, 0] * c[dy, dx, 0]
                    a01 += c[dy, dx, 0] * c[dy, dx, 1]
                    a02 += c[dy, dx, 0] * c[dy, dx, 2]
                    a11 += c[dy, dx, 1] * c[dy, dx, 1]
                    a12 += c[dy, dx, 1] * c[dy, dx, 2]
                    a22 += c[dy, dx, 2] * c[dy, dx, 2]

            a00 /= window_area
            a01 /= window_area
            a02 /= window_area
            a11 /= window_area
            a12 /= window_area
            a22 /= window_area

            det = (
                a00 * a12 * a12
                + a01 * a01 * a22
                + a02 * a02 * a11
                - a00 * a11 * a22
                - 2 * a01 * a02 * a12
            )

            inv_det = 1.0 / det

            # Calculate inverse covariance matrix
            m00 = (a12 * a12 - a11 * a22) * inv_det
            m01 = (a01 * a22 - a02 * a12) * inv_det
            m02 = (a02 * a11 - a01 * a12) * inv_det
            m11 = (a02 * a02 - a00 * a22) * inv_det
            m12 = (a00 * a12 - a01 * a02) * inv_det
            m22 = (a01 * a01 - a00 * a11) * inv_det

            # For each pair ((xi, yi), (xj, yj)) in a (2 r + 1)x(2 r + 1) window
            for dyi in range(2 * r + 1):
                for dxi in range(2 * r + 1):
                    s = c[dyi, dxi, 0]
                    t = c[dyi, dxi, 1]
                    u = c[dyi, dxi, 2]

                    c0 = m00 * s + m01 * t + m02 * u
                    c1 = m01 * s + m11 * t + m12 * u
                    c2 = m02 * s + m12 * t + m22 * u

                    for dyj in range(2 * r + 1):
                        for dxj in range(2 * r + 1):
                            xi = x + dxi - r
                            yi = y + dyi - r
                            xj = x + dxj - r
                            yj = y + dyj - r

                            i = xi + yi * w
                            j = xj + yj * w

                            # Calculate contribution of pixel pair to L_ij
                            temp = (
                                c0 * c[dyj, dxj, 0]
                                + c1 * c[dyj, dxj, 1]
                                + c2 * c[dyj, dxj, 2]
                            )

                            value = (1.0 if (i == j) else 0.0) - (
                                1 + temp
                            ) / window_area

                            dx = xj - xi + 2 * r
                            dy = yj - yi + 2 * r

                            values[i, dy, dx] += value


def cf_laplacian(image, epsilon=1e-7, radius=1, is_known=None):
    """
    This function implements the alpha estimator for closed-form alpha matting as proposed by :cite:`levin2007closed`.

    Parameters
    ------------
    image: numpy.ndarray
       Image with shape :math:`h\\times w \\times 3`
    epsilon: float
       Regularization strength, defaults to :math:`10^{-7}`. Strong regularization improves convergence but results in smoother alpha mattes.
    radius: int
       Radius of local window size, defaults to :math:`1`, i.e. only adjacent pixels are considered.
       The size of the local window is given as :math:`(2 r + 1)^2`, where :math:`r` denotes         the radius. A larger radius might lead to violated color line constraints, but also
       favors further propagation of information within the image.
    is_known: numpy.ndarray
        Binary mask of pixels for which to compute the laplacian matrix.
        Laplacian entries for known pixels will have undefined values.

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian
    """
    h, w, d = image.shape
    n = h * w

    if is_known is None:
        is_known = np.zeros((h, w), dtype=np.bool8)

    is_known = is_known.reshape(h, w)

    # Data for matting laplacian in csr format
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(n * (4 * radius + 1) ** 2, dtype=np.int64)
    values = np.zeros((n, 4 * radius + 1, 4 * radius + 1), dtype=np.float64)

    _cf_laplacian(image, epsilon, radius, values, indices, indptr, is_known)

    L = scipy.sparse.csr_matrix((values.ravel(), indices, indptr), (n, n))

    return L


print("time 2.14: " + datetime.datetime.now().isoformat())


def trimap_split(trimap, flatten=True, bg_threshold=0.1, fg_threshold=0.9):
    """This function splits the trimap into foreground pixels, background pixels, and unknown pixels.

    Foreground pixels are pixels where the trimap has values larger than or equal to `fg_threshold` (default: 0.9).
    Background pixels are pixels where the trimap has values smaller than or equal to `bg_threshold` (default: 0.1).
    Pixels with other values are assumed to be unknown.

    Parameters
    ----------
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times w`
    flatten: bool
        If true np.flatten is called on the trimap

    Returns
    -------
    is_fg: numpy.ndarray
        Boolean array indicating which pixel belongs to the foreground
    is_bg: numpy.ndarray
        Boolean array indicating which pixel belongs to the background
    is_known: numpy.ndarray
        Boolean array indicating which pixel is known
    is_unknown: numpy.ndarray
        Boolean array indicating which pixel is unknown
    bg_threshold: float
        Pixels with smaller trimap values will be considered background.
    fg_threshold: float
        Pixels with larger trimap values will be considered foreground.


    Example
    -------
    >>> import numpy as np
    >>> from pymatting import *
    >>> trimap = np.array([[1,0],[0.5,0.2]])
    >>> is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)
    >>> is_fg
    array([ True, False, False, False])
    >>> is_bg
    array([False,  True, False, False])
    >>> is_known
    array([ True,  True, False, False])
    >>> is_unknown
    array([False, False,  True,  True])
    """
    if flatten:
        trimap = trimap.flatten()

    min_value = trimap.min()
    max_value = trimap.max()

    if min_value < 0.0:
        warnings.warn(
            "Trimap values should be in [0, 1], but trimap.min() is %s." % min_value,
            stacklevel=3,
            )

    if max_value > 1.0:
        warnings.warn(
            "Trimap values should be in [0, 1], but trimap.max() is %s." % min_value,
            stacklevel=3,
            )

    if trimap.dtype not in [np.float32, np.float64]:
        warnings.warn(
            "Unexpected trimap.dtype %s. Are you sure that you do not want to use np.float32 or np.float64 instead?"
            % trimap.dtype,
            stacklevel=3,
            )

    is_fg = trimap >= fg_threshold
    is_bg = trimap <= bg_threshold

    if is_bg.sum() == 0:
        raise ValueError(
            "Trimap did not contain background values (values <= %f)" % bg_threshold
        )

    if is_fg.sum() == 0:
        raise ValueError(
            "Trimap did not contain foreground values (values >= %f)" % fg_threshold
        )

    is_known = is_fg | is_bg
    is_unknown = ~is_known

    return is_fg, is_bg, is_known, is_unknown


print("time 2.15: " + datetime.datetime.now().isoformat())


def sanity_check_image(image):
    """Performs a sanity check for input images. Image values should be in the
    range [0, 1], the `dtype` should be `np.float32` or `np.float64` and the
    image shape should be `(?, ?, 3)`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h \\times w \\times 3`

    Example
    -------
    import numpy as np
     from pymatting import check_image
     image = (np.random.randn(64, 64, 2) * 255).astype(np.int32)
     sanity_check_image(image)
    __main__:1: UserWarning: Expected RGB image of shape (?, ?, 3), but image.shape is (64, 64, 2).
    __main__:1: UserWarning: Image values should be in [0, 1], but image.min() is -933.
    __main__:1: UserWarning: Image values should be in [0, 1], but image.max() is 999.
    __main__:1: UserWarning: Unexpected image.dtype int32. Are you sure that you do not want to use np.float32 or np.float64 instead?

    """

    if len(image.shape) != 3 or image.shape[2] != 3:
        warnings.warn(
            "Expected RGB image of shape (?, ?, 3), but image.shape is %s."
            % str(image.shape),
            stacklevel=3,
            )

    min_value = image.min()
    max_value = image.max()

    if min_value < 0.0:
        warnings.warn(
            "Image values should be in [0, 1], but image.min() is %s." % min_value,
            stacklevel=3,
            )

    if max_value > 1.0:
        warnings.warn(
            "Image values should be in [0, 1], but image.max() is %s." % max_value,
            stacklevel=3,
            )

    if image.dtype not in [np.float32, np.float64]:
        warnings.warn(
            "Unexpected image.dtype %s. Are you sure that you do not want to use np.float32 or np.float64 instead?"
            % image.dtype,
            stacklevel=3,
            )


print("time 2.16: " + datetime.datetime.now().isoformat())


def estimate_alpha_cf(
    image, trimap, preconditioner=None, laplacian_kwargs={}, cg_kwargs={}
):
    """
    Estimate alpha from an input image and an input trimap using Closed-Form Alpha Matting as proposed by :cite:`levin2007closed`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the alpha matte should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w` of the image
    preconditioner: function or scipy.sparse.linalg.LinearOperator
        Function or sparse matrix that applies the preconditioner to a vector (default: ichol)
    laplacian_kwargs: dictionary
        Arguments passed to the :code:`cf_laplacian` function
    cg_kwargs: dictionary
        Arguments passed to the :code:`cg` solver
    is_known: numpy.ndarray
        Binary mask of pixels for which to compute the laplacian matrix.
        Providing this parameter might improve performance if few pixels are unknown.

    Returns
    -------
    alpha: numpy.ndarray
        Estimated alpha matte

    Example
    -------
    >>> from pymatting import *
    >>> image = load_image("data/lemur/lemur.png", "RGB")
    >>> trimap = load_image("data/lemur/lemur_trimap.png", "GRAY")
    >>> alpha = estimate_alpha_cf(
    ...     image,
    ...     trimap,
    ...     laplacian_kwargs={"epsilon": 1e-6},
    ...     cg_kwargs={"maxiter":2000})
    """
    if preconditioner is None:
        preconditioner = ichol

    sanity_check_image(image)

    h, w = trimap.shape[:2]

    is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

    L = cf_laplacian(image, **laplacian_kwargs, is_known=is_known)

    # Split Laplacian matrix L into
    #
    #     [L_U   R ]
    #     [R^T   L_K]
    #
    # and then solve L_U x_U = -R is_fg_K for x where K (is_known) corresponds to
    # fixed pixels and U (is_unknown) corresponds to unknown pixels. For reference, see
    # Grady, Leo, et al. "Random walks for interactive alpha-matting." Proceedings of VIIP. Vol. 2005. 2005.

    L_U = L[is_unknown, :][:, is_unknown]

    R = L[is_unknown, :][:, is_known]

    m = is_fg[is_known]

    x = trimap.copy().ravel()

    x[is_unknown] = cg(L_U, -R.dot(m), M=preconditioner(L_U), **cg_kwargs)

    alpha = np.clip(x, 0, 1).reshape(trimap.shape)

    return alpha
