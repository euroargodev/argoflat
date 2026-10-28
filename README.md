# ARGOFLAT: The Argo dataset as a lossless, analysis-ready, cloud-optimized contiguous ragged array
A solution to enable fast random access to Argo profiles and ML-ready batch sampling from a global aggregate.

‼️ Work in progress ‼️

## Basic concept


## Expect API

Open or connect to an ArgoFlat dataset:
```python
from argoflat import FlatData

fd = FlatData("argoflat_expert.zarr")
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

Profiles selection using standard 0-based array indexing:
```python
ds = fd[100] 
ds = fd[10:20] 
ds = fd[[1, 5, 8]]
ds = fd[:] # Full dataset !
```
