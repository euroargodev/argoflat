import glob as _glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from argopy.utils.checkers import check_wmo, check_cyc

from argoflat.utils import merge_blocks


class FlatData:
    """Argo dataset as a flat (ragged) collection of points, backed by one or several Zarr stores

    Each store has two groups:

    - ``points``: dimension ``N_POINTS``, all valid levels of all profiles, concatenated
    - ``profiles``: dimension ``N_PROF``, profile metadata plus ``row_start`` and ``row_size``
      giving the location of each profile in the ``points`` of *the same store*.

    With several stores, the profile indexes are concatenated in memory into a single one, so
    that all the selection methods work on the whole collection. The ``points`` stay lazy and
    only the stores holding the selected profiles are read. Profiles are numbered globally,
    following the order of the stores: profile ``i`` of store ``k`` has global index
    ``offsets[k] + i``. This numbering changes if the list of stores changes.

    Returned datasets list profiles in *store order* (sorted, no duplicates), which gives
    contiguous, efficient reads.

    Parameters
    ----------
    stores : str or list of str
        One store, a list of stores, or a glob pattern (``"stores/argo_*.zarr"``,
        ``"s3://bucket/argo_*.zarr"``). Stores from a glob are sorted by name.
    broadcast : bool
        Default for the methods returning datasets, see :meth:`float`.
    **storage_options
        Passed to fsspec for remote stores, e.g. ``anon=True``.

    Examples
    --------
    >>> fd = FlatData("argoflat-expert.zarr")
    >>> fd = FlatData("stores/argo_*.zarr")                 # a collection of stores
    >>> fd = FlatData(["a.zarr", "b.zarr"])
    >>> fd = FlatData("s3://bucket/argo_*.zarr", anon=True)

    >>> ds = fd.float(1901514)
    >>> ds = fd.float([1901514, 1902284])
    >>> ds = fd.profile(1901514, [2, 3])
    >>> ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])
    >>> ds = fd.region([-75., 0., 0., 45.])            # no time constraint
    >>> ds = fd[100]; ds = fd[10:20]; ds = fd[[1, 5, 8]]
    """

    # Number of contiguous blocks above which we fall back to a single ``isel`` per store
    # instead of concatenating one slice per block.
    MAX_RUNS = 50

    def __init__(self, stores, broadcast=False, **storage_options):
        self._spec = stores
        self.broadcast = broadcast
        self._kw = {"storage_options": storage_options} if storage_options else {}
        self.refresh()

    # ------------------------------------------------------------------ opening
    def _resolve(self):
        spec = self._spec
        if isinstance(spec, (str, os.PathLike)):
            spec = str(spec)
            if any(c in spec for c in "*?["):
                if "://" in spec:
                    import fsspec

                    fs, p = fsspec.core.url_to_fs(
                        spec, **self._kw.get("storage_options", {})
                    )
                    proto = spec.split("://")[0]
                    stores = sorted(f"{proto}://{x}" for x in fs.glob(p))
                else:
                    stores = sorted(_glob.glob(spec))
            else:
                stores = [spec]
        else:
            stores = [str(s) for s in spec]
        if not stores:
            raise FileNotFoundError(f"No store found for {self._spec!r}")
        return stores

    def refresh(self):
        """(Re)open the stores, e.g. after one has been appended to, or new stores were added"""
        self.stores = self._resolve()
        prfs, self.pts_stores = [], []
        for path in self.stores:
            prf = xr.open_zarr(
                path, group="profiles", **self._kw
            ).load()  # index, fits in memory
            if 'JULD' in prf:
                self._TIME = 'JULD'
            elif 'TIME' in prf:
                self._TIME = 'TIME'
            else:
                raise RuntimeError("No time variable !")
            pts = xr.open_zarr(path, group="points", **self._kw)  # lazy

            n_valid, n_pts = int(prf["row_size"].sum()), pts.sizes["N_POINTS"]
            if n_pts < n_valid:
                raise RuntimeError(
                    f"Corrupted store {path}: 'profiles' references {n_valid} points "
                    f"but 'points' holds only {n_pts}"
                )
            if n_pts > n_valid:
                warnings.warn(
                    f"{path}: 'points' holds {n_pts - n_valid} orphan points not referenced "
                    "by 'profiles' (half-finished append?). They are ignored."
                )
            prfs.append(prf)
            self.pts_stores.append(pts)

        ref_p, ref_l = set(prfs[0].data_vars), set(self.pts_stores[0].data_vars)
        for path, prf, pts in zip(self.stores, prfs, self.pts_stores):
            if set(prf.data_vars) != ref_p or set(pts.data_vars) != ref_l:
                raise ValueError(
                    f"Store {path} has not the same variables as {self.stores[0]}"
                )

        counts = np.array([p.sizes["N_PROF"] for p in prfs])
        self.offsets = np.r_[
            0, np.cumsum(counts)
        ]  # global index of the first profile of each store
        self.prf = (
            prfs[0]
            if len(prfs) == 1
            else xr.concat(prfs, dim="N_PROF", combine_attrs="override")
        )
        self.prf["STORE"] = (
            "N_PROF",
            np.repeat(np.arange(len(prfs)), counts).astype("int32"),
            {"long_name": "index of the store holding the profile"},
        )
        self.N_POINTS = int(self.prf["row_size"].sum())

    # -------------------------------------------------------------- variables
    @property
    def pts(self):
        """The lazy ``points`` dataset (single store only, see ``pts_stores`` otherwise)"""
        if len(self.pts_stores) == 1:
            return self.pts_stores[0]
        raise AttributeError(
            "A collection of stores has no single 'points' dataset, use `pts_stores[i]`"
        )

    @property
    def LEVEL_VARS(self):
        return list(self.pts_stores[0].data_vars)

    @property
    def PROF_VARS(self):
        return [v for v in self.prf.data_vars]
        # return [v for v in self.prf.data_vars if v not in ("row_start", "row_size", "STORE")]

    def _get_profiles(self, sel, broadcast=None):
        """Given sel profile indexes, figure out which point-ranges to read from which stores,
        and read them efficiently."""
        broadcast = self.broadcast if broadcast is None else broadcast

        sel = np.unique(np.atleast_1d(sel).astype("int64"))  # sorted, unique
        if sel.size == 0:
            raise ValueError("No data matching your criteria")
        if sel[0] < 0 or sel[-1] >= self.N_PROF:
            raise IndexError(f"Profile index out of range [0, {self.N_PROF - 1}]")

        start = self.prf["row_start"].values[sel].astype("int64")
        size = self.prf["row_size"].values[sel].astype("int64")
        store = self.prf["STORE"].values[sel]

        # Position in `sel` and level number of each selected point (vectorised)
        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(sel.size), size)
        level = np.arange(size.sum()) - offsets[local]

        # Lazy selection of the points: sel is sorted, so stores are visited in order and the
        # points come out in the same order as the profiles.
        blocks = []  # (store, block_starts, block_ends)
        for k in np.unique(store):
            m = store == k
            block_starts, block_ends = merge_blocks(start[m], start[m] + size[m])
            blocks.append((k, block_starts, block_ends))

        if sum(len(block_starts) for _, block_starts, _ in blocks) <= self.MAX_RUNS:
            parts = [
                self.pts_stores[store].isel(N_POINTS=slice(int(a), int(b)))
                for store, block_starts, block_ends in blocks
                for a, b in zip(block_starts, block_ends)
            ]
        else:
            point_store = store[local]
            parts = []
            for store, _, _ in blocks:
                pm = point_store == store
                parts.append(
                    self.pts_stores[store].isel(N_POINTS=start[local][pm] + level[pm])
                )
        out = (
            parts[0]
            if len(parts) == 1
            else xr.concat(parts, "N_POINTS", combine_attrs="override")
        )

        out = out.assign_coords(N_LEVELS=("N_POINTS", level))

        # Profile variables
        p = self.prf[self.PROF_VARS].isel(N_PROF=sel)
        if broadcast:
            pp = p.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
            out = out.assign({k: pp[k].variable for k in pp.data_vars})
        else:
            p = p.assign_coords(N_PROF=("N_PROF", sel))
            out = out.assign(PROFILE=("N_POINTS", sel[local]))
            out = xr.merge([out, p], combine_attrs="override")

        out.attrs.update(self.pts_stores[0].attrs)
        return out

    def _sel_float(self, WMO):
        WMO = check_wmo(WMO)
        cond = np.isin(self.prf["PLATFORM_NUMBER"].values, WMO)
        return np.flatnonzero(cond)

    def _sel_profile(self, WMO, CYC):
        WMO, CYC = check_wmo(WMO), check_cyc(CYC)
        cond = np.isin(self.prf["PLATFORM_NUMBER"].values, WMO) & np.isin(
            self.prf["CYCLE_NUMBER"].values, CYC
        )
        return np.flatnonzero(cond)

    def _sel_region(self, box):
        lon, lat = self.prf["LONGITUDE"].values, self.prf["LATITUDE"].values

        if box[0] <= box[1]:
            cond = (lon >= box[0]) & (lon < box[1])
        else:  # crosses the antimeridian
            cond = (lon >= box[0]) | (lon < box[1])
        cond &= (lat >= box[2]) & (lat < box[3])

        if len(box) == 6:
            juld = self.prf[self._TIME].values
            cond &= (juld >= pd.to_datetime(box[4]).to_datetime64()) & (
                juld < pd.to_datetime(box[5]).to_datetime64()
            )

        return np.flatnonzero(cond)

    # ------------------------------------------------------------ public API

    def float(self, WMO, broadcast=None):
        """All profiles of one or more floats"""
        sel = self._sel_float(WMO)
        return self._get_profiles(sel, broadcast=broadcast)

    def profile(self, WMO, CYC, broadcast=None):
        """Profiles of one or more floats, for one or more cycle numbers"""
        sel = self._sel_profile(WMO, CYC)
        return self._get_profiles(sel, broadcast=broadcast)

    def region(self, box, broadcast=None):
        """Profiles in a space/time box

        Parameters
        ----------
        box : list
            ``[lon_min, lon_max, lat_min, lat_max]`` or
            ``[lon_min, lon_max, lat_min, lat_max, date_min, date_max]``.
            Intervals are half-open: ``min <= value < max``.
            If ``lon_min > lon_max`` the box is assumed to cross the antimeridian.
        """
        if len(box) not in (4, 6):
            raise ValueError(
                "box must be [lon_min, lon_max, lat_min, lat_max(, date_min, date_max)]"
            )
        sel = self._sel_region(box)
        return self._get_profiles(sel, broadcast=broadcast)

    # ------------------------------------------------------------ properties
    @property
    def profiles(self):
        """Copy of the profile index, to build custom selections (see :meth:`__getitem__`)

        Examples
        --------
        >>> p = fd.profiles
        >>> idx = np.flatnonzero((p.DATA_MODE == b"D").values)
        >>> ds = fd[idx]
        """
        return self.prf.copy()

    @property
    def N_PROF(self):
        return self.prf.sizes["N_PROF"]

    @property
    def n_stores(self):
        return len(self.stores)

    @property
    def domain(self):
        """Space/time domain of the dataset, as a dictionary of (min, max) tuples"""

        def mm(name):
            v = self.prf[name].values
            if v.size == 0 or (
                np.issubdtype(v.dtype, np.floating) and np.isnan(v).all()
            ):
                return (None, None)
            if np.issubdtype(v.dtype, np.datetime64):
                v = v[~np.isnat(v)]
                return (v.min(), v.max()) if v.size else (None, None)
            return (np.nanmin(v), np.nanmax(v))

        return {"lon": mm("LONGITUDE"), "lat": mm("LATITUDE"), "time": mm(self._TIME)}

    # --------------------------------------------------------------- dunders
    def __len__(self):
        return self.N_PROF

    def store_of(self, sel):
        """Index of the store holding each of the given profiles (global indices)"""
        return self.prf["STORE"].values[np.asarray(sel)]

    def __repr__(self):
        d = self.domain
        fmt = lambda a, b, f="{:.2f}": (
            "N/A" if a is None else f"[{f.format(a)}, {f.format(b)}]"
        )
        t = (
            "N/A"
            if d["time"][0] is None
            else f"[{str(d['time'][0])[:16]}, {str(d['time'][1])[:16]}]"
        )
        return "\n".join(
            [
                "<argopy.flatdata>",
                f"Stores: {self.n_stores}"
                + (
                    f" ({self.stores[0]})"
                    if self.n_stores == 1
                    else f" ({self.stores[0]} ... {self.stores[-1]})"
                ),
                f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})",
                f"Broadcast: {self.broadcast}",
                "Domain:",
                f" - Longitude: {fmt(*d['lon'])}",
                f" - Latitude: {fmt(*d['lat'])}",
                f" - Time: {t}",
            ]
        )

    def __getitem__(self, ii):
        n = np.arange(self.N_PROF)
        if isinstance(ii, (int, np.integer)):
            sel = n[[ii]]  # handles negative indices, raises IndexError if out of range
        elif isinstance(ii, slice):
            sel = n[ii]
        elif isinstance(ii, (list, tuple, np.ndarray)):
            ii = np.asarray(ii)
            sel = n[ii]  # works for integer arrays and boolean masks
        else:
            raise NotImplementedError(
                f"This indexing method ({type(ii)}) is not implemented"
            )
        return self._get_profiles(sel)
