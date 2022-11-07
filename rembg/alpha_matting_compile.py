from numba.pycc import CC

cc = CC('alpha_matting_precompiled')
cc.verbose = True

import datetime

print("time 2.4: " + datetime.datetime.now().isoformat())
from numba import njit, prange

print("time 2.5: " + datetime.datetime.now().isoformat())
import numpy as np

@njit()
@cc.export("resize_nearest_multichannel", "void(f4[:, :, :], f4[:, :, :])")
def resize_nearest_multichannel(dst, src):
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


@njit()
@cc.export("resize_nearest", "void(f4[:, :], f4[:, :])")
def resize_nearest(dst, src):
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


@njit()
@cc.export("estimate_fb_ml", "Tuple((f4[:, :, :], f4[:, :, :]))(f4[:, :, :], f4[:, :], f4, i4, i4, i4, f4)", )
def estimate_fb_ml(
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

        resize_nearest_multichannel(image, input_image)
        resize_nearest(alpha, input_alpha)

        F = np.empty((h, w, depth), dtype=dtype)
        B = np.empty((h, w, depth), dtype=dtype)

        resize_nearest_multichannel(F, F_prev)
        resize_nearest_multichannel(B, B_prev)

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


@njit()
@cc.export("backsub_L_csc_inplace", "void(f8[:], i8[:], i8[:], f8[:], i8)")
def backsub_L_csc_inplace(data, indices, indptr, x, n):
    for j in range(n):
        k = indptr[j]
        L_jj = data[k]
        temp = x[j] / L_jj

        x[j] = temp

        for k in range(indptr[j] + 1, indptr[j + 1]):
            i = indices[k]
            L_ij = data[k]

            x[i] -= L_ij * temp


@njit()
@cc.export("backsub_LT_csc_inplace", "void(f8[:], i8[:], i8[:], f8[:], i8)")
def backsub_LT_csc_inplace(data, indices, indptr, x, n):
    for i in range(n - 1, -1, -1):
        s = x[i]

        for k in range(indptr[i] + 1, indptr[i + 1]):
            j = indices[k]
            L_ji = data[k]
            s -= L_ji * x[j]

        k = indptr[i]
        L_ii = data[k]

        x[i] = s / L_ii





@njit()
@cc.export("ichol", "i8(i8, f8[:], i8[:], i8[:], f8[:], i8[:], i8[:], f8, f8, i8, f8, b1)")
def ichol(
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

@njit()
@cc.export("cf_laplacian", "void(f8[:, :, :], f8, i8, f8[:, :, :], i8[:], i8[:], b1[:, :])")
def cf_laplacian(image, epsilon, r, values, indices, indptr, is_known):
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

            if np.all(is_known[y - r: y + r + 1, x - r: x + r + 1]):
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


if __name__ == "__main__":
    cc.compile()
