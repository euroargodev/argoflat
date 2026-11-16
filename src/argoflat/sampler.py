import numpy as np
from argoflat.utils import merge_blocks

try:
    import torch
    from torch.utils.data import IterableDataset

except Exception as e:
    IterableDataset = object
    pass

class _FlatSamplerDataset(IterableDataset):
    """Picklable wrapper: one instance per worker, holds a ref to the sampler."""

    def __init__(self, sampler: "FlatSampler"):
        self._sampler = sampler

    def __iter__(self):
        for batch in self._sampler:
            yield {
                k: torch.as_tensor(v) if v.dtype.kind in "biuf" else v
                for k, v in batch.items()
            }

    def __len__(self):
        return len(self._sampler)


class FlatSampler:
    """Random, memory-bounded batch sampler over a :class:`FlatData` store

    Fully random access to a chunked store is slow (one chunk read per profile).
    Instead, shuffling is done in two levels:

    1. the selected profiles are cut into *blocks* of ``block_size`` consecutive profiles
       (contiguous in the store, so each block is read with a few slice reads), and the
       block order is shuffled at each epoch;
    2. ``mix_blocks`` blocks are loaded together into a buffer, whose profiles are shuffled
       before being cut into batches.

    A larger ``mix_blocks`` gives a better shuffle (profiles from more distinct floats and
    regions in each batch) at the cost of memory:
    ``mix_blocks * block_size * max_levels * n_variables * 4`` bytes (float32).

    Parameters
    ----------
    fd : FlatData
    variables : list of str, optional
        Level variables to return, default ``["PRES", "TEMP", "PSAL"]``.
    profile_vars : list of str, optional
        Per-profile variables to return, default ``["LATITUDE", "LONGITUDE", "JULD"]``.
        Datetimes are converted to float days since 1970-01-01.
    sel : array_like of int, optional
        Profile indices to sample from (e.g. from :meth:`FlatData.split`). Default: all.
    batch_size : int
    max_levels : int, optional
        Profiles are padded (or truncated) to this number of levels.
        Default is the largest profile in ``sel``, which can be wasteful if there are outliers.
    shuffle : bool
        If False, profiles are returned in store order (useful for validation/inference).
    seed : int
        Together with the epoch number (see :meth:`set_epoch`), it fully determines the order.
    block_size, mix_blocks : int
        See above.
    drop_last : bool
        Drop the last incomplete batch of each shard.
    rank, world_size : int
        For distributed training: this process is ``rank`` out of ``world_size``.
        Torch ``DataLoader`` workers are detected automatically and get their own share.

    Yields
    ------
    dict
        - one ``(batch, max_levels)`` array per level variable (padded with NaN, or 0 for non-float);
        - ``mask``: ``(batch, max_levels)`` boolean, True where a level exists;
        - ``row_size``: number of levels in each profile (before truncation);
        - one ``(batch,)`` array per profile variable;
        - ``PROFILE``: ``(batch,)`` index of each profile in the store, to trace samples back
          (``fd[batch["PROFILE"]]``).
    """

    def __init__(self, fd, variables=None, profile_vars=("LATITUDE", "LONGITUDE", "JULD"), sel=None,
                 batch_size=64, max_levels=None, shuffle=True, seed=0, block_size=256, mix_blocks=8,
                 drop_last=False, rank=0, world_size=1):
        self.fd = fd
        if "JULD" in profile_vars:
            profile_vars = list(profile_vars)
            profile_vars[profile_vars.index("JULD")] = fd._TIME
            profile_vars = tuple(profile_vars)

        self.variables = list(variables) if variables is not None else [v for v in ("PRES", "TEMP", "PSAL") if
                                                                        v in fd.pts]
        self.profile_vars = list(profile_vars)
        missing = [v for v in self.variables if v not in fd.pts] + [v for v in self.profile_vars if v not in fd.prf]
        if missing:
            raise ValueError(f"Variables not found in the store: {missing}")

        self.sel = np.arange(fd.N_PROF) if sel is None else np.unique(np.asarray(sel, dtype="int64"))
        if self.sel.size == 0:
            raise ValueError("Empty selection")
        if self.sel[0] < 0 or self.sel[-1] >= fd.N_PROF:
            raise IndexError(f"Profile index out of range [0, {fd.N_PROF - 1}]")

        self.max_levels = int(max_levels or fd.prf["row_size"].values[self.sel].max())
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.block_size = int(block_size)
        self.mix_blocks = int(mix_blocks)
        self.drop_last = drop_last
        self.rank, self.world_size = rank, world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        """Change the shuffle at each epoch (same epoch and seed give the same order)"""
        self.epoch = int(epoch)

    def __len__(self):
        """Number of batches over the whole selection (all shards), approximately"""
        n = self.sel.size
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    # -------------------------------------------------------------- internals
    def _shard(self):
        """(shard_id, n_shards) combining distributed rank and torch DataLoader workers"""
        worker_id, n_workers = 0, 1
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
            if info is not None:
                worker_id, n_workers = info.id, info.num_workers
        except ImportError:
            pass
        return self.rank * n_workers + worker_id, self.world_size * n_workers

    def _read_block(self, sel):
        """Read profiles `sel` (sorted) into padded numpy arrays"""
        fd, L = self.fd, self.max_levels
        start = fd.prf["row_start"].values[sel].astype("int64")
        size = fd.prf["row_size"].values[sel].astype("int64")
        run_starts, run_ends = merge_blocks(start, start + size)

        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(sel.size), size)
        level = np.arange(size.sum()) - offsets[local]
        keep = level < L

        out = {}
        for v in self.variables:
            flat = np.concatenate([
                fd.pts[v].isel(N_POINTS=slice(int(s), int(e))).values for s, e in zip(run_starts, run_ends)
            ])
            fill = np.nan if np.issubdtype(flat.dtype, np.floating) else 0
            padded = np.full((sel.size, L), fill, dtype=flat.dtype)
            padded[local[keep], level[keep]] = flat[keep]
            out[v] = padded

        out["mask"] = np.arange(L)[None, :] < size[:, None]
        out["row_size"] = size
        for v in self.profile_vars:
            x = fd.prf[v].values[sel]
            if np.issubdtype(x.dtype, np.datetime64):
                x = (x - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            out[v] = x
        out["PROFILE"] = sel
        return out

    @staticmethod
    def _cat(a, b):
        return b if a is None else {k: np.concatenate([a[k], b[k]]) for k in a}

    @staticmethod
    def _take(d, idx):
        return {k: v[idx] for k, v in d.items()}

    def __iter__(self):
        bs = self.batch_size
        shard, n_shards = self._shard()

        # same block order in all shards, then each shard keeps its own share
        rng = np.random.default_rng([self.seed, self.epoch])
        blocks = [self.sel[i:i + self.block_size] for i in range(0, self.sel.size, self.block_size)]
        order = rng.permutation(len(blocks)) if self.shuffle else np.arange(len(blocks))
        order = order[shard::n_shards]
        rng = np.random.default_rng([self.seed, self.epoch, shard])  # independent row shuffle per shard

        carry = None
        for g in range(0, len(order), self.mix_blocks):
            buf = carry
            for b in order[g:g + self.mix_blocks]:
                buf = self._cat(buf, self._read_block(blocks[b]))
            n = len(buf["PROFILE"])
            if self.shuffle:
                buf = self._take(buf, rng.permutation(n))
            n_full = n // bs
            for i in range(n_full):
                yield self._take(buf, slice(i * bs, (i + 1) * bs))
            carry = self._take(buf, slice(n_full * bs, n)) if n_full * bs < n else None

        if carry is not None and not self.drop_last:
            yield carry

    # ------------------------------------------------------------------ torch
    def torch_dataset(self):
        """Wrap the sampler into a :class:`torch.utils.data.IterableDataset`

        Numeric arrays are converted to tensors. Use ``batch_size=None`` in the ``DataLoader``
        since batching is done here.

        Examples
        --------
        ..code-block:: python

            from torch.utils.data import DataLoader

            ds = fd.sampler(sel=train, batch_size=128).torch_dataset()
            loader = DataLoader(ds, batch_size=None, num_workers=4)

        Notes
        -----
        Workers get disjoint shares of the blocks. Call ``sampler.set_epoch(epoch)`` before
        iterating the loader at each epoch (this works when workers are re-created at each epoch,
        i.e. ``persistent_workers=False``, the default).
        """
        return _FlatSamplerDataset(self)
        # import torch
        #
        # sampler = self
        #
        # class _Dataset(torch.utils.data.IterableDataset):
        #     def __iter__(self):
        #         for batch in sampler:
        #             yield {k: torch.as_tensor(v) if v.dtype.kind in "biuf" else v for k, v in batch.items()}
        #
        #     def __len__(self):
        #         return len(sampler)
        #
        # return _Dataset()
