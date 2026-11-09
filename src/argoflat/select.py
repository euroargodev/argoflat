import xarray as xr
import numpy as np
import pandas as pd

from argopy.utils.casting import to_list
from argopy.utils.checkers import check_wmo, check_cyc


def float(prf: xr.Dataset, WMO) -> np.ndarray:
    """Select profile indices fo one or more floats (WMO numbers).

    Parameters
    ----------
    int | list[int]
    """
    WMO = check_wmo(WMO)
    cond = np.isin(prf["PLATFORM_NUMBER"].values, WMO)
    return np.flatnonzero(cond)

def profile(prf: xr.Dataset, WMO, CYC) -> np.ndarray:
    """Select profile indices for one or more (WMO, CYCLE_NUMBER) pairs.

    Parameters
    ----------
    WMO: int | list[int]
    CYC: int | list[int]
    """
    WMO, CYC = check_wmo(WMO), check_cyc(CYC)
    cond = np.isin(prf["PLATFORM_NUMBER"].values, WMO) & np.isin(prf["CYCLE_NUMBER"].values, CYC)
    return np.flatnonzero(cond)

def region(prf: xr.Dataset, box, time_var: str) -> np.ndarray:
    """Select profile indices inside the ``[lon, lat]`` / ``[lon, lat, time]`` box

    Parameters
    ----------
    box : list
        ``[lon_min, lon_max, lat_min, lat_max]`` or
        ``[lon_min, lon_max, lat_min, lat_max, date_min, date_max]``.
        Intervals are half-open: ``min <= value < max``.
        If ``lon_min > lon_max`` the box is assumed to cross the antimeridian.
        ``date_min`` and ``date_max`` are strings.
    """
    if len(box) not in (4, 6):
        raise ValueError(
            "box must be [lon_min, lon_max, lat_min, lat_max(, date_min, date_max)]"
        )

    lon, lat = prf["LONGITUDE"].values, prf["LATITUDE"].values

    if box[0] <= box[1]:
        cond = (lon >= box[0]) & (lon < box[1])
    else:  # crosses the antimeridian
        cond = (lon >= box[0]) | (lon < box[1])
    cond &= (lat >= box[2]) & (lat < box[3])

    if len(box) == 6:
        juld = prf[time_var].values
        cond &= (juld >= pd.to_datetime(box[4]).to_datetime64()) & (
                juld < pd.to_datetime(box[5]).to_datetime64()
        )

    return np.flatnonzero(cond)

def year(prf: xr.Dataset, year, time_var: str) -> np.ndarray:
    """Select profile indices for one more given year(s).

    Parameters
    ----------
    year: int | list[int]
    """
    year = to_list(year)
    cond = np.isin(prf[time_var].dt.year, year)
    return np.flatnonzero(cond)