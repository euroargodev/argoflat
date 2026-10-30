# ARGOFLAT: The Argo dataset as a lossless, analysis-ready, cloud-optimized contiguous ragged array
A simple solution to enable fast random access to Argo profiles and ML-ready batch sampling from a global aggregate.

‼️ Work in progress ‼️

Our full Argo dataset is currently build with the entire GDAC and pressure, temperature and salinity data (BGC will come).

The dataset:
- has: 20_565 floats, 3_418_235 P/T/S profiles and 1_360_094_341 measurements points.
- is about ~10.8 GB, as one or more zarr stores.
- is available for testing in 3 versions:
  - As a single zarr store (~10.8 GB)
  - As a collection of 11 zarr stores (~1 GB) with about 2000 floats each
  - As a collection of 21 zarr stores (~500 MB) with about 1000 floats each 


## Basic concept

<img src="https://raw.githubusercontent.com/euroargodev/argoflat/master/docs/_static/ARGOFLAT_basic_concept.jpeg" alt="argoflat" width="800"/>

## User API

Open or connect to an ArgoFlat dataset:
```python
from argoflat import FlatData

fd = FlatData("stores/argoflat_expert_000.zarr")
fd = FlatData("stores_1000/argoflat_expert_0*.zarr")
fd = FlatData("stores_2000/argoflat_expert_0*.zarr")

# fd = FlatData("argoflat_standard.zarr")
# fd = FlatData("argoflat_research.zarr")
# fd = FlatData("s3://bucket/argoflat_*.zarr", anon=True)
```

Use profiles selectors similar to [Argopy access point syntax](https://argopy.readthedocs.io/en/latest/user-guide/fetching-argo-data/data_selection.html):
```python
ds = fd.region([-75., 0., 0., 45.])
ds = fd.region([-75., 0., 0., 45., '20000101', '20201231'])

ds = fd.float(1901514)
ds = fd.float([1901514, 1902284])

ds = fd.profile(1901514, [2, 3])
ds = fd.profile([1901514, 1902284], 3)
ds = fd.profile([1901514, 1902284], [2, 3])
```

Or use profiles selection with standard 0-based indexing:
```python
ds = fd[100] 
ds = fd[10:20] 
ds = fd[[1, 5, 8]]
ds = fd[:] # Full dataset !
```

The xarray dataset that is return has all profiles meta-data loaded in memory (eg: LATITUDE, LONGITUDE, etc ...) and all measurements in 1_000_000 points chunks (eg: PRES, TEMP, etc...).