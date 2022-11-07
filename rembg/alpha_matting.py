import numpy as np
import scipy.sparse
import warnings

import alpha_matting_precompiled

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

    foreground, background = alpha_matting_precompiled.estimate_fb_ml(
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
        alpha_matting_precompiled.backsub_L_csc_inplace(Lv, Lr, Lp, x, n)
        alpha_matting_precompiled.backsub_LT_csc_inplace(Lv, Lr, Lp, x, n)
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
        nnz = alpha_matting_precompiled.ichol(
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

    alpha_matting_precompiled.cf_laplacian(image, epsilon, radius, values, indices, indptr, is_known)

    L = scipy.sparse.csr_matrix((values.ravel(), indices, indptr), (n, n))

    return L


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
