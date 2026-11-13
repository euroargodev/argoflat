import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
from argopy.utils.casting import to_list


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


def get_profiles(sel, prf, pts_stores, broadcast=None, MAX_RUNS=None)-> xr.Dataset:
    """Select profiles and points according to N_PROF dim indexing

    Read points and profile variables for a selection of profile indices.

    Parameters
    ----------
    sel : array_like of int
        Profile indices to load. Sorted and de‑duplicated internally.
    prf:
    pts_stores:
    broadcast : bool or None
        If ``None``, uses :attr:`broadcast`. If ``True``, profile variables are
        repeated along ``N_POINTS``; otherwise they stay on ``N_PROF`` and each
        point carries a ``PROFILE`` pointer and an ``N_LEVELS`` coordinate.
    MAX_RUNS:

    Returns
    -------
    :class:`xarray.Dataset`
        The selected points plus profile metadata.

    Raises
    ------
    ValueError
        If ``sel`` is empty.
    IndexError
        If any index is outside ``[0, N_PROF)``.
    """
    broadcast = broadcast or False
    N_PROF = prf.sizes["N_PROF"]

    sel = np.unique(np.atleast_1d(sel).astype("int64"))  # sorted, unique
    if sel.size == 0:
        raise ValueError("No data matching your criteria")
    if sel[0] < 0 or sel[-1] >= N_PROF:
        raise IndexError(f"Profile index out of range [0, {N_PROF - 1}]")

    start = prf["row_start"].values[sel].astype("int64")
    size = prf["row_size"].values[sel].astype("int64")
    store = prf["store"].values[sel]
    blocks, local, level = plan_reads(start, size, store, sel)

    if (
        MAX_RUNS is not None
        and sum(len(block_starts) for _, block_starts, _ in blocks) <= MAX_RUNS
    ):
        parts = []
        for store_idx, block_starts, block_ends in blocks:
            for a, b in zip(block_starts, block_ends):
                parts.append(
                    pts_stores[store_idx].isel(N_POINTS=slice(int(a), int(b)))
                )
    else:
        point_store = store[local]
        parts = []
        for store_idx, _, _ in blocks:
            pm = point_store == store_idx
            parts.append(
                pts_stores[store_idx].isel(
                    N_POINTS=start[local][pm] + level[pm]
                )
            )
    out = (
        parts[0]
        if len(parts) == 1
        else xr.concat(parts, "N_POINTS", combine_attrs="override")
    )

    out = out.assign_coords(N_LEVELS=("N_POINTS", level))

    # Profile variables
    prf = prf.isel(N_PROF=sel)
    if broadcast:
        pp = prf.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
        out = out.assign({k: pp[k].variable for k in pp.data_vars})
    else:
        prf = prf.assign_coords(N_PROF=("N_PROF", sel))
        out = out.assign(
            puid=("N_POINTS", sel[local]),
        )
        out = xr.merge([out, prf], combine_attrs="override")
        out["puid"].attrs.update(
            {
                "long_name": "Profile index (CF instance variable)",
                "cf_role": "profile_id",
            },
        )
        out['N_PROF'] = np.arange(len(out['N_PROF']))

    out.attrs.update(pts_stores[0].attrs)
    out.attrs.update({"featureType": "profile"})
    return out

def zarr_size(path):
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )

# def zarr_size(path: str) -> int:
#     """Total size in bytes of a zarr store.
#     Works with local paths, s3://, and http(s):// URLs.
#     """
#     fs, path = fsspec.core.url_to_fs(path)
#     if fs.protocol in ['file', 's3']:
#         return fs.du(path)
#     return None


def sizeof_fmt(num, suffix="B"):
    if num is None:
        return ''
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def flatsizeof(fd: "FlatData", fmt=False):
    # "points" group:
    # Theoretical size, if the entire object would be loaded in memory:
    pts_nbytes = 0
    for s in fd.pts_stores:
        for v in s.data_vars:
            pts_nbytes += s[v].nbytes

    # "profiles" group:
    # Theoretical size, if the entire object would be loaded in memory:
    prf_nbytes = 0
    for v in fd.prf.data_vars:
        prf_nbytes += fd.prf[v].nbytes

    # Size of zarr archives on disk:
    stores_size_disk = []
    for s in fd.stores:
        stores_size_disk.append(zarr_size(s))

    data = {
        "disk": sum(stores_size_disk),
        "disk_ave": int(sum(stores_size_disk) / len(stores_size_disk)),
        "ram_max": pts_nbytes + prf_nbytes,
        "ram_max_pts": pts_nbytes,
        "ram_max_prf": prf_nbytes,
    }

    if fmt:
        for key, val in data.items():
            data[key] = sizeof_fmt(val)

    return data


def pressure_bin_stats(ds, edges, pres="PRES", dim="N_POINTS", variables=None):
    """Bin points by pressure and return per-bin stats for every variable.

    Parameters
    ----------
    ds : xr.Dataset
        Point-layout dataset with dimension ``dim`` (default ``"N_POINTS"``).
    edges : array-like
        Bin edges, e.g. ``np.arange(0, 2001, 100)`` (length ``n_bins + 1``).
    pres : str, optional
        The pressure (binning) variable, default ``"PRES"``.
    dim : str, optional
        The points dimension, default ``"N_POINTS"``.
    variables : list of str, optional
        Variables to reduce. Default ``None`` = every numeric data variable.

    Returns
    -------
    pd.DataFrame
        One row per bin. Columns: ``z`` (bin center), ``z_p`` (mean pressure),
        ``n`` (count), and for each variable ``v`` a pair ``v_mean`` / ``v_std``.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)

    # Assign each point to a bin (out-of-range -> NaN, dropped by groupby)
    bin_idx = np.digitize(ds[pres].values, edges, right=False) - 1
    bin_idx = np.where((bin_idx >= 0) & (bin_idx < n_bins), bin_idx, np.nan)
    ds = ds.assign_coords(pbin=(dim, bin_idx))
    pbin = ds["pbin"]

    # Which variables to reduce
    if variables is None:
        variables = [v for v in ds.data_vars
                     if np.issubdtype(ds[v].dtype, np.number)]
        # if v != pres and np.issubdtype(ds[v].dtype, np.number)]
    variables = to_list(variables)

    data = {
        "z_p": ds[pres].groupby(pbin).mean(),
        "n": ds[pres].groupby(pbin).count(),
    }
    for v in variables:
        data[f"{v}_mean"] = ds[v].groupby(pbin).mean()
        data[f"{v}_std"] = ds[v].groupby(pbin).std()

    out = xr.Dataset(data)
    out = out.reset_index('pbin')
    out["z"] = [int(centers[v]) for v in out["pbin"].astype(int).values]
    order = ["z", "z_p", "n"] + [c for v in variables for c in (f"{v}_mean", f"{v}_std")]
    return out[order]


def time_bin_stats(ds, freq='M', time_var='TIME', dim='N_POINTS',
                   variables=None, start=None, end=None):
    """Bin points by month or day and return per-bin stats for every variable.

    Parameters
    ----------
    ds : xr.Dataset
        Point-layout dataset with dimension ``dim`` (default ``"N_POINTS"``)
        and a datetime64 time variable ``time_var``.
    freq : str
        ``'M'`` (or ``'MS'`` / ``'monthly'``) for monthly bins,
        ``'D'`` (or ``'daily'``) for daily bins.
    time_var : str, optional
        Name of the time variable, default ``"TIME"``.
    dim : str, optional
        Points dimension, default ``"N_POINTS"``.
    variables : list of str, optional
        Variables to reduce. Default ``None`` = every numeric data variable.
    start, end : datetime-like, optional
        If given, only points with ``start <= time <= end`` are binned.

    Returns
    -------
    xr.Dataset
        One row per bin. Vars: ``tbin`` (sort key), ``time`` (label),
        ``n`` (count), and for each ``v`` a pair ``v_mean`` / ``v_std``.
    """
    import pandas as pd

    monthly = freq.upper() in ('M', 'MS', 'MONTH', 'MONTHLY')

    tidx = pd.DatetimeIndex(ds[time_var].values)

    # Optional [start, end] range restriction
    if start is not None or end is not None:
        s = pd.Timestamp(start) if start is not None else pd.Timestamp.min
        e = pd.Timestamp(end) if end is not None else pd.Timestamp.max
        valid = (tidx >= s) & (tidx <= e)
    else:
        valid = np.ones(tidx.size, dtype=bool)

    # Sortable integer bin key + human-readable label
    if monthly:
        key = tidx.year * 100 + tidx.month          # 202001, 202002, ...
        fmt = '%Y-%m'
    else:
        key = ((tidx - tidx.min()).astype('timedelta64[ns]').astype('int64')
               // 10**9 // 86400)                    # integer days since min
        fmt = '%Y-%m-%d'

    labels = tidx.strftime(fmt)
    label_map = dict(zip(key[valid], labels[valid]))

    bin_key = np.where(valid, key.astype(float), np.nan)   # out-of-range -> NaN
    ds = ds.assign_coords(tbin=(dim, bin_key))
    tbin = ds['tbin']

    # Which variables to reduce
    if variables is None:
        variables = [v for v in ds.data_vars
                     if v != time_var and np.issubdtype(ds[v].dtype, np.number)]
    variables = to_list(variables)

    data = {'n': ds[time_var].groupby(tbin).count()}
    for v in variables:
        data[f'{v}_mean'] = ds[v].groupby(tbin).mean()
        data[f'{v}_std'] = ds[v].groupby(tbin).std()

    out = xr.Dataset(data)
    out = out.reset_index('tbin')
    out['time'] = [label_map[int(k)] for k in out['tbin'].astype(int).values]
    order = ['tbin', 'time', 'n'] + [c for v in variables for c in (f'{v}_mean', f'{v}_std')]
    return out[order]


def gridpoints(time, p, z, time_grid, pres_grid, method='nearest'):
    from scipy.interpolate import griddata

    # --- numeric time axis (seconds since a reference) ---
    t0 = pd.Timestamp('2000-01-01')  # any fixed origin
    t = (pd.DatetimeIndex(time) - t0).total_seconds()

    ok = np.isfinite(t) & np.isfinite(p) & np.isfinite(z)

    # --- target grid ---
    Tsec, P = np.meshgrid((time_grid - t0).total_seconds(), pres_grid, indexing='ij')  # both (N_TIME, N_PRES)

    # --- interpolate ---
    zi = griddata((p[ok], t[ok]), z[ok], (P, Tsec), method=method)  # (N_TIME, N_PRES)

    out = xr.DataArray(zi,
                       dims=('time', 'pres'),
                       coords={'time': time_grid, 'pres': pres_grid})

    return out