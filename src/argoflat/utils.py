import numpy as np
import xarray as xr
import glob
import os


def resolve_stores(
    spec: str | list[str], storage_options: dict | None = None
) -> list[str]:
    """Resolve a store specification to an ordered list of store paths.

    Parameters
    ----------
    spec : str or list of str
        A single path, a list of paths, or a glob pattern. Glob patterns may be local (``"stores/argo_*.zarr"``)
        or remote (``"s3://bucket/argo_*.zarr"``, resolved via fsspec).
    storage_options : dict, optional
        Passed to fsspec for remote globs, e.g. ``{"anon": True}``.

    Returns
    -------
    list of str
        The resolved store paths, sorted by name.

    Raises
    ------
    FileNotFoundError
        If the pattern resolves to no stores.
    """
    storage_options = storage_options or {}
    if isinstance(spec, (str, os.PathLike)):
        spec = str(spec)
        if any(c in spec for c in "*?["):
            if "://" in spec:
                import fsspec

                fs, p = fsspec.core.url_to_fs(spec, **storage_options)
                proto = spec.split("://")[0]
                stores = sorted(f"{proto}://{x}" for x in fs.glob(p))
            else:
                stores = sorted(glob.glob(spec))
        else:
            stores = [spec]
    else:
        stores = [str(s) for s in spec]
    if not stores:
        raise FileNotFoundError(f"No store found for {spec!r}")
    return stores


def merge_blocks(
    interval_starts: np.ndarray, interval_ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge consecutive or overlapping [start, end) intervals into contiguous blocks.

    Both inputs must be 1-D arrays of half-open intervals sorted in ascending
    order by start. Intervals that touch (``end`` of one == ``start`` of the
    next) or overlap are collapsed into a single contiguous block.

    Parameters
    ----------
    interval_starts : np.ndarray
        Start of each interval (inclusive), ascending.
    interval_ends : np.ndarray
        End of each interval (exclusive), same length as ``interval_starts``.

    Returns
    -------
    block_starts, block_ends : tuple of np.ndarray
        Start and (exclusive) end of each merged contiguous block.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        starts = np.array([4, 10, 100])
        ends   = np.array([10, 21, 151])

        merge_blocks(starts, ends)
        # (array([4, 100]), array([21, 151]))

    """
    # Index of the first interval that begins each new
    # block: always 0, plus every position where a gap separates two intervals
    # (i.e. the previous interval's end != this interval's start).
    block_break_indices = np.flatnonzero(interval_starts[1:] != interval_ends[:-1]) + 1

    # The block starts are the first interval overall plus each interval after a break.
    block_starts = interval_starts[np.r_[0, block_break_indices]]

    # The block ends are the interval just before each break plus the very last one.
    block_ends = interval_ends[np.r_[block_break_indices - 1, interval_ends.size - 1]]

    return block_starts, block_ends


def plan_reads(
    start: np.ndarray, size: np.ndarray, store: np.ndarray, sel: np.ndarray
) -> tuple[list, np.ndarray, np.ndarray]:
    """Compute the lazy‑read plan for a selection of profiles.

    Given the per‑profile point layout of the selected profiles, this function determines:

    - how the points are grouped into contiguous blocks per store,
    - and which selected profile and level number each point belongs to.

    Parameters
    ----------
    start : np.ndarray
        ``row_start`` of each selected profile (int64).
    size : np.ndarray
        ``row_size`` of each selected profile (int64), same length as ``start``.
    store : np.ndarray
        Store index of each selected profile, same length as ``start``.
    sel : np.ndarray
        Selected (sorted, unique) profile indices; length = number of profiles.

    Returns
    -------
    blocks : list of (int, np.ndarray, np.ndarray)
        Per store (ascending): ``(store_idx, block_starts, block_ends)``, the merged
        contiguous ``[start, end)`` point ranges to read.
    local : np.ndarray
        Profile position in ``sel`` for each point (length ``size.sum()``).
    level : np.ndarray
        0‑based level within the profile for each point, same order as ``local``.

    Notes
    -----
    ``sel`` is sorted, so stores are visited in order and points come out in profile order.
    """
    # Position in `sel` and level number of each selected point (vectorised)
    offsets = np.cumsum(size) - size
    local = np.repeat(np.arange(sel.size), size)
    level = np.arange(size.sum()) - offsets[local]

    # Lazy selection of the points: sel is sorted, so stores are visited in order and the
    # points come out in the same order as the profiles.
    blocks = []  # (store, block_starts, block_ends)
    for k in np.unique(store):
        m = store == k
        block_starts, block_end = merge_blocks(start[m], start[m] + size[m])
        blocks.append((k, block_starts, block_end))
    return blocks, local, level
