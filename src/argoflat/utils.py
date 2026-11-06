import numpy as np


# def merge_blocks(starts, ends):
#     """Merge consecutive [start, end) intervals (given in ascending order) into contiguous runs"""
#     breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1  # where a gap appears
#     run_starts = starts[np.r_[0, breaks]] # np.r_: a compact way to write np.concatenate
#     run_ends = ends[np.r_[breaks - 1, len(ends) - 1]]
#     return run_starts, run_ends

def merge_blocks(interval_starts: np.ndarray, interval_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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