"""Classical two-dimensional phase-unwrapping algorithms.

The implementations accept a 2-D wrapped phase array in radians and return a
2-D unwrapped phase array.  They intentionally have no access to ground truth.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np
from scipy.fft import dctn, idctn


Array = np.ndarray


def wrap_phase(value: Array) -> Array:
    """Wrap radians to [-pi, pi)."""
    return np.angle(np.exp(1j * value))


def _validate(wrapped: Array) -> Array:
    wrapped = np.asarray(wrapped, dtype=np.float64)
    if wrapped.ndim != 2:
        raise ValueError(f"expected a 2-D phase image, got shape {wrapped.shape}")
    if not np.isfinite(wrapped).all():
        raise ValueError("wrapped phase contains NaN or infinity")
    return wrap_phase(wrapped)


def unwrap_itoh_2d(wrapped: Array) -> Array:
    """Sequential Itoh unwrapping, first across rows and then columns."""
    wrapped = _validate(wrapped)
    return np.unwrap(np.unwrap(wrapped, axis=1), axis=0)


def _pixel_reliability(wrapped: Array) -> Array:
    """Estimate reliability from wrapped second differences (higher is better)."""
    padded = np.pad(wrapped, 1, mode="edge")
    center = padded[1:-1, 1:-1]

    horizontal = wrap_phase(padded[1:-1, :-2] - center) - wrap_phase(
        center - padded[1:-1, 2:]
    )
    vertical = wrap_phase(padded[:-2, 1:-1] - center) - wrap_phase(
        center - padded[2:, 1:-1]
    )
    diagonal_a = wrap_phase(padded[:-2, :-2] - center) - wrap_phase(
        center - padded[2:, 2:]
    )
    diagonal_b = wrap_phase(padded[:-2, 2:] - center) - wrap_phase(
        center - padded[2:, :-2]
    )
    second_difference = (
        horizontal**2 + vertical**2 + diagonal_a**2 + diagonal_b**2
    )
    return 1.0 / np.sqrt(second_difference + 1e-12)


def unwrap_quality_guided(wrapped: Array) -> Array:
    """Quality-guided maximum-spanning-tree phase unwrapping.

    Pixels are added with Prim's algorithm using edge reliability as the
    priority.  Phase increments are always circular differences, so the tree
    defines a globally consistent integration path without consulting labels.
    """
    wrapped = _validate(wrapped)
    height, width = wrapped.shape
    reliability = _pixel_reliability(wrapped)
    unwrapped = np.empty_like(wrapped)
    visited = np.zeros((height, width), dtype=bool)

    seed_flat = int(np.argmax(reliability))
    seed = divmod(seed_flat, width)
    unwrapped[seed] = wrapped[seed]
    visited[seed] = True
    frontier: list[tuple[float, int, int, int, int]] = []

    def push_neighbors(row: int, col: int) -> None:
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and not visited[next_row, next_col]
            ):
                edge_reliability = reliability[row, col] + reliability[next_row, next_col]
                heapq.heappush(
                    frontier,
                    (-float(edge_reliability), row, col, next_row, next_col),
                )

    push_neighbors(*seed)
    while frontier:
        _, row, col, next_row, next_col = heapq.heappop(frontier)
        if visited[next_row, next_col]:
            continue
        unwrapped[next_row, next_col] = unwrapped[row, col] + wrap_phase(
            wrapped[next_row, next_col] - wrapped[row, col]
        )
        visited[next_row, next_col] = True
        push_neighbors(next_row, next_col)

    if not visited.all():  # defensive: a rectangular 4-neighbour grid is connected
        raise RuntimeError("quality-guided traversal did not visit every pixel")
    return unwrapped


def unwrap_least_squares(wrapped: Array) -> Array:
    """Unweighted least-squares unwrapping via a Neumann Poisson solve."""
    wrapped = _validate(wrapped)
    height, width = wrapped.shape

    gradient_x = wrap_phase(wrapped[:, 1:] - wrapped[:, :-1])
    gradient_y = wrap_phase(wrapped[1:, :] - wrapped[:-1, :])
    divergence = np.zeros_like(wrapped)
    divergence[:, :-1] -= gradient_x
    divergence[:, 1:] += gradient_x
    divergence[:-1, :] -= gradient_y
    divergence[1:, :] += gradient_y

    frequency_y = np.arange(height, dtype=np.float64)[:, None]
    frequency_x = np.arange(width, dtype=np.float64)[None, :]
    eigenvalues = (
        2.0 - 2.0 * np.cos(np.pi * frequency_y / height)
        + 2.0
        - 2.0 * np.cos(np.pi * frequency_x / width)
    )
    transformed = dctn(divergence, type=2, norm="ortho")
    transformed[0, 0] = 0.0
    eigenvalues[0, 0] = 1.0
    solution = idctn(transformed / eigenvalues, type=2, norm="ortho")

    # The Neumann solution has arbitrary additive constant. Anchor it to the
    # same wrapped value as the observation without using ground truth.
    anchor = float(wrapped[0, 0] - wrap_phase(solution[0, 0]))
    return solution + anchor


def _laplacian_neumann(value: Array) -> Array:
    """Five-point Laplacian with replicated (zero-normal-gradient) boundaries."""
    padded = np.pad(value, 1, mode="edge")
    return (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - 4.0 * value
    )


def _inverse_laplacian_dct(source: Array) -> Array:
    """Solve Laplacian(u)=source under Neumann boundary conditions by DCT."""
    height, width = source.shape
    ky = np.arange(height, dtype=np.float64)[:, None]
    kx = np.arange(width, dtype=np.float64)[None, :]
    eigenvalues = (
        2.0 * np.cos(np.pi * ky / height)
        + 2.0 * np.cos(np.pi * kx / width)
        - 4.0
    )
    transformed = dctn(source, type=2, norm="ortho")
    transformed[0, 0] = 0.0
    eigenvalues[0, 0] = 1.0
    return idctn(transformed / eigenvalues, type=2, norm="ortho")


def unwrap_dct_schofield(wrapped: Array) -> Array:
    """Schofield-type DCT phase unwrapping from the complex-phase Laplacian.

    The Laplacian is recovered from sin/cos of the wrapped observation, then a
    Neumann Poisson problem is solved by DCT.  Integer cycle rounding restores
    exact modulo-2pi consistency with the input observation.
    """
    wrapped = _validate(wrapped)
    sine, cosine = np.sin(wrapped), np.cos(wrapped)
    source = cosine * _laplacian_neumann(sine) - sine * _laplacian_neumann(cosine)
    smooth = _inverse_laplacian_dct(source)
    cycles = np.rint((smooth - wrapped) / (2.0 * np.pi))
    solution = wrapped + 2.0 * np.pi * cycles
    # Fix the otherwise arbitrary global integer cycle without using labels.
    solution += 2.0 * np.pi * np.rint((wrapped[0, 0] - solution[0, 0]) / (2.0 * np.pi))
    return solution


METHODS: dict[str, Callable[[Array], Array]] = {
    "itoh": unwrap_itoh_2d,
    "quality_guided_mst": unwrap_quality_guided,
    "least_squares": unwrap_least_squares,
    "dct_schofield": unwrap_dct_schofield,
}
