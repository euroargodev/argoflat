import warnings

import numpy as np
import xarray as xr

from argoflat.utils import resolve_stores, plan_reads, flatsizeof
from argoflat import select


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

    Examples
    --------
    Open a single store, a glob collection, an explicit list, or a remote store:

    .. code-block:: python

        fd = FlatData("argoflat_expert.zarr")
        fd = FlatData("stores/argo_*.zarr")              # a collection of stores
        fd = FlatData(["a.zarr", "b.zarr"])
        fd = FlatData("s3://bucket/argo_*.zarr", anon=True)

    Every selector returns an :class:`xarray.Dataset`:

    .. code-block:: python

        ds = fd.float(1901514)
        ds = fd.float([1901514, 1902284])
        ds = fd.profile(1901514, [2, 3])
        ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])
        ds = fd.region([-75., 0., 0., 45.])              # no time constraint
        ds = fd[100]; ds = fd[10:20]; ds = fd[[1, 5, 8]]
    """

    def __init__(
        self,
        stores: str | list[str],
        broadcast: bool = False,
        MAX_RUNS: int | None = 200,
        dry: bool = False,
        **storage_options,
    ):
        """Initialize the flat collection of points and open its stores.

        Parameters
        ----------
        stores : str or list of str
            One store, a list of stores, or a glob pattern (``"stores/argo_*.zarr"``,
        ``"s3://bucket/argo_*.zarr"``). Stores from a glob are sorted by name.
        broadcast : bool, default False
            Default for the methods returning datasets, see :meth:`float`.
        MAX_RUNS : int or None, default 200
            Number of contiguous blocks above which the read falls back to a single
            :meth:`xarray.Dataset.isel` per store instead of concatenating one slice per block.
            ``None`` disables the threshold (always use the slice strategy).
        **storage_options
            Passed to fsspec for remote stores, e.g. ``anon=True``.

        Notes
        -----
        This also calls :meth:`refresh`, so the profile index is loaded into memory at instanciation time.
        """
        self.N_POINTS = None
        """Global number of measurements"""

        self.pts_stores: list[xr.Dataset] = None
        """List of lazy points :class:`xr.Dataset`"""

        self.stores = None
        """Ordered list of store paths"""

        self._spec = stores
        self.broadcast = broadcast
        self._kw = {"storage_options": storage_options} if storage_options else {}
        self.MAX_RUNS = MAX_RUNS
        self.dry = dry
        self.refresh()

    # ------------------------------------------------------------------ opening
    def refresh(self) -> None:
        """(Re)open the stores, e.g. after one has been appended to, or new stores were added.

        Loads each store's ``profiles`` index into memory, keeps the ``points`` group
        lazy, validates that all stores expose the same variables, and builds the
        in‑memory global profile index (``self.prf``) with a ``STORE`` variable and
        per‑store ``offsets``.

        Raises
        ------
        RuntimeError
            If a store has no time variable, or its ``profiles`` reference more points
            than its ``points`` group holds.
        ValueError
            If stores do not all share the same set of variables.
        """
        self.stores = resolve_stores(self._spec, self._kw.get("storage_options", {}))
        prfs, self.pts_stores = [], []
        for path in self.stores:
            prf = xr.open_zarr(
                path, group="profiles", **self._kw
            ).load()  # index, fits in memory
            if "JULD" in prf:
                self._TIME = "JULD"
            elif "TIME" in prf:
                self._TIME = "TIME"
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
        self._offsets = np.r_[0, np.cumsum(counts)]
        """Global index of the first profile of each store"""

        self.prf = (
            prfs[0]
            if len(prfs) == 1
            else xr.concat(prfs, dim="N_PROF", combine_attrs="override")
        )
        """Description of profiles as :class:`xr.Dataset`"""

        self.prf["row_size"].attrs = {
            "standard_name": "number_of_levels",
            "sample_dimension": "N_POINTS",  # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_contiguous_ragged_array_representation
            "long_name": "Number of valid levels per profile (CF count variable)",
        }
        self.prf["row_start"].attrs = {
            "long_name": "Offset of each profile into the concatenated points.",
            "comment": "Redundant to row_size = exclusive cumulative sum of row_size",
        }
        self.prf["store"] = (
            "N_PROF",
            np.repeat(np.arange(len(prfs)), counts).astype("int32"),
            {"long_name": "index of the store holding the profile"},
        )
        self.N_POINTS = int(self.prf["row_size"].sum())
        """Global number of measurements"""

    # -------------------------------------------------------------- variables
    @property
    def pts(self) -> xr.Dataset:
        """The lazy ``points`` dataset (single‑store collections only).

        Returns
        -------
        :class:`xr.Dataset`

        Raises
        ------
        AttributeError
            If the collection holds more than one store; use ``pts_stores[i]`` instead.
        """
        if len(self.pts_stores) == 1:
            return self.pts_stores[0]
        raise AttributeError(
            "A collection of stores has no single 'points' dataset, use `pts_stores[i]` instead."
        )

    @property
    def LEVEL_VARS(self) -> list[str]:
        """List of the point‑level (``N_POINTS``) variables of the dataset."""
        return list(self.pts_stores[0].data_vars)

    @property
    def PROF_VARS(self) -> list[xr.DataArray]:
        """List of the profile (``N_PROF``) variables carried by the index."""
        return [v for v in self.prf.data_vars]
        # return [v for v in self.prf.data_vars if v not in ("row_start", "row_size", "store")]

    def get_profiles(self, sel, broadcast=None) -> xr.Dataset:
        """Read points and profile variables for a selection of profile indices.

        Parameters
        ----------
        sel : array_like of int
            Profile indices to load. Sorted and de‑duplicated internally.
        broadcast : bool or None
            If ``None``, uses :attr:`broadcast`. If ``True``, profile variables are
            repeated along ``N_POINTS``; otherwise they stay on ``N_PROF`` and each
            point carries a ``PROFILE`` pointer and an ``N_LEVELS`` coordinate.

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
        broadcast = self.broadcast if broadcast is None else broadcast

        sel = np.unique(np.atleast_1d(sel).astype("int64"))  # sorted, unique
        if sel.size == 0:
            raise ValueError("No data matching your criteria")
        if sel[0] < 0 or sel[-1] >= self.N_PROF:
            raise IndexError(f"Profile index out of range [0, {self.N_PROF - 1}]")

        start = self.prf["row_start"].values[sel].astype("int64")
        size = self.prf["row_size"].values[sel].astype("int64")
        store = self.prf["store"].values[sel]
        blocks, local, level = plan_reads(start, size, store, sel)

        if (
            self.MAX_RUNS is not None
            and sum(len(block_starts) for _, block_starts, _ in blocks) <= self.MAX_RUNS
        ):
            parts = []
            for store_idx, block_starts, block_ends in blocks:
                for a, b in zip(block_starts, block_ends):
                    parts.append(
                        self.pts_stores[store_idx].isel(N_POINTS=slice(int(a), int(b)))
                    )
        else:
            point_store = store[local]
            parts = []
            for store_idx, _, _ in blocks:
                pm = point_store == store_idx
                parts.append(
                    self.pts_stores[store_idx].isel(
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
        prf = self.prf[self.PROF_VARS].isel(N_PROF=sel)
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

        out.attrs.update(self.pts_stores[0].attrs)
        out.attrs.update({"featureType": "profile"})
        return out

    # ------------------------------------------------------------ public API

    def float(
        self,
        WMO: int | list[int],
        broadcast: None | bool = None,
        dry: bool | None = None,
    ) -> xr.Dataset:
        """All profiles of one or more floats.

        Parameters
        ----------
        WMO : int or list of int
            Argo float WMO number(s).
        broadcast : bool or None
            See :meth:`get_profiles`.

        Returns
        -------
        :class:`xarray.Dataset`

        Examples
        --------
        .. code-block:: python

            ds = fd.float(1901514)
            ds = fd.float([1901514, 1902284])
        """
        sel = select.float(self.prf, WMO)
        if (dry is not None and dry) or self.dry:
            return len(sel)
        return self.get_profiles(sel, broadcast=broadcast)

    def profile(
        self,
        WMO: int | list[int],
        CYC: int | list[int],
        broadcast: None | bool = None,
        dry: bool | None = None,
    ) -> xr.Dataset:
        """Profiles of one or more floats, for one or more cycle numbers.

        Parameters
        ----------
        WMO : int or list of int
            Argo float WMO number(s).
        CYC : int or list of int
            Cycle number(s).
        broadcast : bool or None
            See :meth:`get_profiles`.

        Returns
        -------
        :class:`xarray.Dataset`

        Examples
        --------
        .. code-block:: python

            ds = fd.profile(1901514, [2, 3])
        """
        sel = select.profile(self.prf, WMO, CYC)
        if (dry is not None and dry) or self.dry:
            return len(sel)
        return self.get_profiles(sel, broadcast=broadcast)

    def region(
        self, box: list, broadcast: None | bool = None, dry: bool | None = None
    ) -> xr.Dataset:
        """Profiles in a space/time box.

        Parameters
        ----------
        box : list
            ``[lon_min, lon_max, lat_min, lat_max]`` or
            ``[lon_min, lon_max, lat_min, lat_max, date_min, date_max]``.
            Intervals are half-open: ``min <= value < max``.
            If ``lon_min > lon_max`` the box is assumed to cross the antimeridian.
            Note that ``date_min`` and ``date_max`` are strings.
        """
        sel = select.region(self.prf, box, self._TIME)
        if (dry is not None and dry) or self.dry:
            return len(sel)
        return self.get_profiles(sel, broadcast=broadcast)

    def year(
        self,
        year: int | list[int],
        broadcast: None | bool = None,
        dry: bool | None = None,
    ) -> xr.Dataset:
        """Profiles for one or more year(s).

        Parameters
        ----------
        year : int or list of int
            Observation year(s).
        broadcast : bool or None
            See :meth:`get_profiles`.

        Returns
        -------
        xarray.Dataset

        Examples
        --------
        .. code-block:: python

            ds = fd.year(2025)
            ds = fd.year([2024, 2025])
        """
        sel = select.year(self.prf, year, self._TIME)
        if (dry is not None and dry) or self.dry:
            return len(sel)
        return self.get_profiles(sel, broadcast=broadcast)

    @property
    def N_PROF(self) -> int:
        """Total number of profiles across all stores."""
        return self.prf.sizes["N_PROF"]

    @property
    def n_stores(self) -> int:
        """Number of stores in the collection."""
        return len(self.stores)

    @property
    def domain(self) -> dict:
        """Space/time domain of the dataset as a dict of ``(min, max)`` tuples.

        Returns
        -------
        dict
            Keys ``"lon"``, ``"lat"``, ``"time"``, each mapping to a ``(min, max)``
            tuple (or ``(None, None)`` when the extent is unknown).
        """

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
    def __len__(self) -> int:
        """Number of profiles (``N_PROF``), so ``len(fd)`` works."""
        return self.N_PROF

    def store_of(self, sel: np.ndarray[tuple[int]]) -> np.ndarray[tuple[int]]:
        """Index of the store holding each of the given profiles.

        Parameters
        ----------
        sel : int or array_like of int
            Profile index or indices.

        Returns
        -------
        :class:`np.ndarray`
            Store index (into ``pts_stores``) for each profile.
        """
        return self.prf["store"].values[np.asarray(sel)]

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
        sizeof = flatsizeof(self, fmt=True)
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
                f"Bytes:",
                f" - On disk: {sizeof['disk']}",
                f" - In memory if loaded: {sizeof['ram_max']}",
                "Domain:",
                f" - Longitude: {fmt(*d['lon'])}",
                f" - Latitude: {fmt(*d['lat'])}",
                f" - Time: {t}",
            ]
        )

    def __getitem__(self, ii) -> xr.Dataset:
        """Index the dataset by profile index or indices.

        Supports a single int, a slice, or a list/tuple/ndarray of indices (or a
        boolean mask), e.g. ``fd[100]``, ``fd[10:20]``, ``fd[[1, 5, 8]]``.

        Parameters
        ----------
        ii : int, slice, list, tuple, or np.ndarray
            Profile index or indices.

        Returns
        -------
        :class:`xarray.Dataset`

        Raises
        ------
        NotImplementedError
            If an unsupported indexing type is used.
        """
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
        return self.get_profiles(sel)
