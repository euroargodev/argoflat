import numpy as np
import pandas as pd
import xarray as xr
from functools import cached_property
from argopy.utils.checkers import check_wmo, check_cyc, is_indexbox
# from argoflat.static import VariablesShort as Variables
from argoflat.static import Variables



class FlatData0:
    """Argo dataset as a flat collection of points

    Examples
    --------
    ..code-block:: python

        from argopy.utils.arco import FlatData

        fd = FlatData("argoflat-expert.zarr")

        ds = fd.float(1901514)
        ds = fd.float([1901514, 1902284])
        ds = fd.profile(1901514, [2, 3])
        ds = fd.profile([1901514, 1902284], 3)
        ds = fd.profile([1901514, 1902284], [2, 3])
        ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])

        fg[100]
    """

    def __init__(self, store, broadcast=False):
        self.store = store
        self.broadcast = broadcast
        self.prf = xr.open_zarr(store, group="profiles").load()  # fit in memory, even for the full expert dataset
        self.pts = xr.open_zarr(store, group="points")  # lazy

    def _get_profiles(self, sel, broadcast=None):
        if len(sel) == 0:
            raise ValueError('No data matching your criteria')
        broadcast = self.broadcast if broadcast is None else broadcast

        sel = np.atleast_1d(sel)
        start = self.prf.row_start.values[sel]
        size = self.prf.row_size.values[sel]

        # vectorised construction of the point indices (no python loop)
        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(len(sel)), size)  # position in `sel` for each point
        level = np.arange(size.sum()) - offsets[local]  # level number within its profile
        idx = start[local] + level

        out = self.pts.isel(N_POINTS=idx)  # still lazy
        p = self.prf.isel(N_PROF=sel).drop_vars(["row_start", "row_size"])

        if broadcast:
            # every profile variable repeated along N_POINTS
            pp = p.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
            out = out.assign({k: pp[k] for k in pp.data_vars})
        else:
            # profile variables stay on N_PROF; points carry a pointer to them
            p = p.assign_coords(N_PROF=sel)
            out = out.assign(PROFILE=("N_POINTS", sel[local]))
            out = xr.merge([out, p])

        out = out.assign_coords(N_POINTS=("N_POINTS", level))

        # for v in out:
        #     if v in attrs:
        #         out[v].attrs = attrs[v]

        out = out[[v for v in Variables.PROF_VARS + Variables.LEVEL_VARS]]
        return out

    def float(self, WMO, broadcast=None):
        WMO = check_wmo(WMO)
        filters = []
        for wmo in WMO:
            filters.append(self.prf['PLATFORM_NUMBER'] == wmo)
        cond = np.logical_or.reduce(filters)
        sel = np.flatnonzero(cond)
        ds = self._get_profiles(sel, broadcast=broadcast)
        return ds

    def profile(self, WMO, CYC, broadcast=None):
        WMO = check_wmo(WMO)
        CYC = check_cyc(CYC)
        sel = []
        for wmo in WMO:
            for cyc in CYC:
                f = np.logical_and.reduce([self.prf['PLATFORM_NUMBER'] == wmo, self.prf['CYCLE_NUMBER'] == cyc])
                [sel.append(i) for i in np.flatnonzero(f)]
        ds = self._get_profiles(sel, broadcast=broadcast)
        return ds

    def region(self, box, broadcast=None):
        is_indexbox(box)
        filters = []
        filters.append(self.prf['LONGITUDE'] >= box[0])
        filters.append(self.prf['LONGITUDE'] < box[1])
        filters.append(self.prf['LATITUDE'] >= box[2])
        filters.append(self.prf['LATITUDE'] < box[3])
        filters.append(self.prf['JULD'] >= pd.to_datetime(box[4]))
        filters.append(self.prf['JULD'] < pd.to_datetime(box[5]))
        cond = np.logical_and.reduce(filters)
        sel = np.flatnonzero(cond)
        ds = self._get_profiles(sel, broadcast=broadcast)
        return ds

    @property
    def N_PROF(self):
        return len(self.prf['N_PROF'])

    @property
    def N_POINTS(self):
        return self.prf['row_size'].sum().item()

    @property
    def domain(self):
        """Space/time domain of the dataset

        This is different from a usual argopy ``box`` because dates are in :class:`numpy.datetime64` format.
        """
        return [
            np.nanmin(self.prf["LONGITUDE"].values),
            np.nanmax(self.prf["LONGITUDE"].values),
            np.nanmin(self.prf["LATITUDE"].values),
            np.nanmax(self.prf["LATITUDE"].values),
            None,
            None,
            np.nanmin(self.prf['JULD'].values),
            np.nanmax(self.prf['JULD'].values),
        ]

    def __repr__(self):
        summary = ["<argopy.flatdata>"]
        summary.append(f"Store: {self.store}")
        summary.append(f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})")
        summary.append(f"Broadcast: {self.broadcast}")
        domain = self.domain
        summary.append("Domain:")
        summary.append(f" - Longitude: [{domain[0]:.2f}, {domain[1]:.2f}]")
        summary.append(f" - Latitude: [{domain[2]:.2f}, {domain[3]:.2f}]")
        summary.append(f" - Time: [{str(domain[6])[0:16]}, {str(domain[7])[0:16]}]")
        return "\n".join(summary)




class FlatData:
    """Argo dataset as a flat collection of points

    Examples
    --------
    ..code-block:: python

        from argopy.utils.arco import FlatData

        fd = FlatData("argoflat-expert.zarr")

        ds = fd.float(1901514)
        ds = fd.float([1901514, 1902284])
        ds = fd.profile(1901514, [2, 3])
        ds = fd.profile([1901514, 1902284], 3)
        ds = fd.profile([1901514, 1902284], [2, 3])
        ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])

        fg[100]
    """

    def __init__(self, store, broadcast=False, **storage_options):
        self.store = store
        self.broadcast = broadcast

        # storage_options will allow to connect to a S3 store
        kw = dict(storage_options=storage_options) if storage_options else {}
        self.prf = xr.open_zarr(store, group="profiles", **kw).load()  # fit in memory, even for the full expert dataset
        self.pts = xr.open_zarr(store, group="points", **kw)  # lazy

    def _get_profiles(self, sel, broadcast=None):
        if len(sel) == 0:
            raise ValueError('No data matching your criteria')
        broadcast = self.broadcast if broadcast is None else broadcast

        sel = np.atleast_1d(sel)
        start = self.prf.row_start.values[sel]
        size = self.prf.row_size.values[sel]

        # vectorised construction of the point indices (no python loop)
        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(len(sel)), size)  # position in `sel` for each point
        level = np.arange(size.sum()) - offsets[local]  # level number within its profile
        idx = start[local] + level

        out = self.pts.isel(N_POINTS=idx)  # still lazy
        p = self.prf.isel(N_PROF=sel).drop_vars(["row_start", "row_size"])

        if broadcast:
            # every profile variable repeated along N_POINTS
            pp = p.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
            out = out.assign({k: pp[k] for k in pp.data_vars})
        else:
            # profile variables stay on N_PROF; points carry a pointer to them
            p = p.assign_coords(N_PROF=sel)
            out = out.assign(PROFILE=("N_POINTS", sel[local]))
            out = xr.merge([out, p])

        # out = out.assign_coords(N_POINTS=("N_POINTS", level))
        out = out.assign_coords(SAMPLE=("N_POINTS", level))

        # for v in out:
        #     if v in attrs:
        #         out[v].attrs = attrs[v]

        out = out[[v for v in Variables.PROF_VARS + Variables.LEVEL_VARS]]
        return out

    def float(self, WMO, broadcast=None):
        WMO = check_wmo(WMO)
        cond = np.isin(self.prf["PLATFORM_NUMBER"].values, WMO)
        sel = np.flatnonzero(cond)
        return self._get_profiles(sel, broadcast=broadcast)

    def profile(self, WMO, CYC, broadcast=None):
        WMO, CYC = check_wmo(WMO), check_cyc(CYC)
        cond = (np.isin(self.prf["PLATFORM_NUMBER"].values, WMO)
                & np.isin(self.prf["CYCLE_NUMBER"].values, CYC))
        return self._get_profiles(np.flatnonzero(cond), broadcast=broadcast)

        return ds

    def region(self, box, broadcast=None):
        is_indexbox(box)
        filters = []
        filters.append(self.prf['LONGITUDE'] >= box[0])
        filters.append(self.prf['LONGITUDE'] < box[1])
        filters.append(self.prf['LATITUDE'] >= box[2])
        filters.append(self.prf['LATITUDE'] < box[3])
        filters.append(self.prf['JULD'] >= pd.to_datetime(box[4]))
        filters.append(self.prf['JULD'] < pd.to_datetime(box[5]))
        cond = np.logical_and.reduce(filters)
        sel = np.flatnonzero(cond)
        return self._get_profiles(sel, broadcast=broadcast)

    @cached_property
    def N_PROF(self):
        return self.prf.sizes["N_PROF"]

    @cached_property
    def N_POINTS(self):
        return self.pts.sizes["N_POINTS"]

    @cached_property
    def domain(self):
        """Space/time domain of the dataset

        This is different from a usual argopy ``box`` because dates are in :class:`numpy.datetime64` format.
        """
        return [
            np.nanmin(self.prf["LONGITUDE"].values),
            np.nanmax(self.prf["LONGITUDE"].values),
            np.nanmin(self.prf["LATITUDE"].values),
            np.nanmax(self.prf["LATITUDE"].values),
            None,
            None,
            np.nanmin(self.prf['JULD'].values),
            np.nanmax(self.prf['JULD'].values),
        ]

    def __repr__(self):
        summary = ["<argopy.flatdata>"]
        summary.append(f"Store: {self.store}")
        summary.append(f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})")
        summary.append(f"Broadcast: {self.broadcast}")
        domain = self.domain
        summary.append("Domain:")
        summary.append(f" - Longitude: [{domain[0]:.2f}, {domain[1]:.2f}]")
        summary.append(f" - Latitude: [{domain[2]:.2f}, {domain[3]:.2f}]")
        summary.append(f" - Time: [{str(domain[6])[0:16]}, {str(domain[7])[0:16]}]")
        return "\n".join(summary)

    def __getitem__(self, ii):
        # if type(ii) == int:
        #     return self._get_profiles([ii])
        # elif type(ii) == slice:
        #     return self._get_profiles(np.arange(self.N_PROF)[ii])
        # elif isinstance(ii, (list, np.ndarray)):
        #     return self._get_profiles(np.asarray(ii))
        # raise NotImplementedError(f"This indexing method ({type(ii)}) is not implemented")
        n = np.arange(self.N_PROF)
        if isinstance(ii, (int, np.integer)):
            sel = n[[ii]]  # handles negative indices, raises IndexError if out of range
        elif isinstance(ii, slice):
            sel = n[ii]
        elif isinstance(ii, (list, tuple, np.ndarray)):
            ii = np.asarray(ii)
            sel = n[ii]  # works for integer arrays and boolean masks
        else:
            raise NotImplementedError(f"This indexing method ({type(ii)}) is not implemented")
        return self._get_profiles(sel)
