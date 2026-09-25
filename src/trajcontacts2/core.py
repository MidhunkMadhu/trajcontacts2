"""Core contact-calculation routines for trajcontacts.

The public entry point is :func:`compute_contact_counts`, which returns, for
every requested residue pair, the number of frames in which the minimum
inter-residue heavy-atom distance falls at or below a cutoff.

The definition of a contact is unchanged from trajcontacts 0.1.x: a pair of
residues is in contact in a given frame when the shortest distance between any
two of their non-hydrogen atoms is <= ``cutoff``.  What changed is how that
number is obtained:

* distances come from one vectorised :func:`mdtraj.compute_distances` call per
  block of residue pairs, reduced to per-pair minima exactly as
  :func:`mdtraj.compute_contacts` (``scheme="closest-heavy"``) does, instead of
  one ``compute_distances`` call per residue pair;
* frames are processed in chunks so peak memory stays bounded;
* a cheap, *exact* geometric prefilter discards pairs that cannot possibly be
  in contact within a chunk, which for a folded protein removes the large
  majority of the O(N^2) pair list.

The prefilter is a lower bound, never an approximation.  For residues i and j,

    min_heavy_distance(i, j) >= centroid_distance(i, j) - radius(i) - radius(j)

where ``radius`` is the distance from a residue's heavy-atom centroid to its
outermost heavy atom.  A pair is only discarded when that lower bound exceeds
the cutoff in every frame of the chunk, so no contact can be missed.

Continuous ("semi-Gaussian") contacts
-------------------------------------
:func:`compute_continuous_contacts` replaces the 0/1 indicator by a smooth
weight of the same per-frame minimum heavy-atom distance ``d``::

    K(d) = 1                                   if d <= c
    K(d) = exp(-d^2 / 2s^2) / exp(-c^2 / 2s^2) otherwise
         = exp(-(d^2 - c^2) / 2s^2)

with ``s = sqrt((c^2 - d_max^2) / (2 ln k))``, so that ``K(d_max) = k``
(trajcontacts 1.0.1 defaults: c = 0.45 nm, d_max = 0.8 nm, k = 1e-5, giving
s = 0.13784 nm; Westerlund et al. use a fixed s = 0.138 nm). The weights are
summed over frames and divided by the number of frames. The prefilter then
uses the kernel's support instead of the cutoff: the distance beyond which a
legacy-rounded weight is exactly 0, or the unrounded weight is below a
tolerance. :func:`compute_contacts_both` produces both kinds of result from
one distance pass.
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from dataclasses import dataclass, field

import mdtraj as md
import numpy as np
from mdtraj.core import element as md_element

__all__ = [
    "ContactResult",
    "ContinuousContactResult",
    "ResidueInfo",
    "compute_contact_counts",
    "compute_contacts_both",
    "compute_continuous_contacts",
    "continuous_sigma",
    "describe_residues",
    "heavy_atom_indices",
    "kernel_support",
    "make_pairs",
    "semi_gaussian_kernel",
]

# Bytes of scratch space the chunking heuristics aim to stay under.
_DEFAULT_MEMORY_BUDGET = 2_000_000_000

# np.around(K, 4) maps every K below half a unit of the 4th decimal to 0.
_LEGACY_ZERO_LEVEL = 0.5e-4

# Extra prefilter radius (nm) on top of the kernel support. It absorbs the
# float32 rounding of the prefilter's lower bound; with the default kernel it
# is worth about 2e-6 of weight at the legacy support, far from the rounding
# boundary.
_SUPPORT_MARGIN_NM = 1e-3


@dataclass(frozen=True)
class ResidueInfo:
    """Identity of one residue, carrying the labels the input file used."""

    index: int          # 0-based index within the selected subsystem
    chain: str          # chain identifier from the topology
    res_seq: int        # residue number as written in the input file
    name: str           # residue name, e.g. "ALA"
    segment: str = ""   # segment id, when the topology provides one

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.name}{self.res_seq}"


@dataclass
class ContactResult:
    """Outcome of a contact calculation."""

    pairs: np.ndarray                    # (n_pairs, 2) residue indices
    counts: np.ndarray                   # (n_pairs,) frames in contact
    n_frames: int
    residues: list = field(default_factory=list)   # list[ResidueInfo]
    cutoff_nm: float = 0.45
    n_pairs_considered: int = 0
    n_pairs_prefiltered: int = 0

    @property
    def fractions(self) -> np.ndarray:
        if self.n_frames == 0:
            return np.zeros_like(self.counts, dtype=float)
        return self.counts / float(self.n_frames)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def count_matrix(self) -> np.ndarray:
        """Symmetric (n_residues, n_residues) matrix of frame counts."""
        n = self.n_residues
        mat = np.zeros((n, n), dtype=np.int64)
        if len(self.pairs):
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            mat[i, j] = self.counts
            mat[j, i] = self.counts
        return mat

    def fraction_matrix(self) -> np.ndarray:
        if self.n_frames == 0:
            return np.zeros((self.n_residues, self.n_residues), dtype=float)
        return self.count_matrix() / float(self.n_frames)

    def adjacency_matrix(self, fraction_cutoff: float) -> np.ndarray:
        """Unweighted adjacency matrix: 1 where fraction >= ``fraction_cutoff``."""
        return (self.fraction_matrix() >= fraction_cutoff).astype(np.int64)


@dataclass
class ContinuousContactResult:
    """Outcome of a continuous (semi-Gaussian) contact calculation."""

    pairs: np.ndarray                    # (n_pairs, 2) residue indices
    sums: np.ndarray                     # (n_pairs,) kernel weight summed over frames
    n_frames: int
    residues: list = field(default_factory=list)   # list[ResidueInfo]
    cutoff_nm: float = 0.45
    sigma_nm: float = 0.0
    support_nm: float = float("inf")     # weights beyond this were not evaluated
    legacy_rounding: bool = False
    n_pairs_considered: int = 0
    n_pairs_prefiltered: int = 0

    @property
    def means(self) -> np.ndarray:
        """Frame-averaged weight of each pair, in [0, 1]."""
        if self.n_frames == 0:
            return np.zeros_like(self.sums, dtype=float)
        return self.sums / float(self.n_frames)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def matrix(self) -> np.ndarray:
        """Symmetric (n_residues, n_residues) matrix of mean weights, zero diagonal."""
        n = self.n_residues
        mat = np.zeros((n, n), dtype=np.float64)
        if len(self.pairs):
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            means = self.means
            mat[i, j] = means
            mat[j, i] = means
        return mat

    def condensed(self) -> np.ndarray:
        """Upper triangle of :meth:`matrix` in ``np.triu_indices(n, 1)`` order.

        This is the layout of ``scipy.spatial.distance.squareform``.
        """
        n = self.n_residues
        return self.matrix()[np.triu_indices(n, 1)]

    def adjacency_matrix(self, threshold: float) -> np.ndarray:
        """Unweighted adjacency matrix: 1 where the mean weight >= ``threshold``."""
        return (self.matrix() >= threshold).astype(np.int64)


def heavy_atom_indices(topology) -> list:
    """Per-residue arrays of non-hydrogen atom indices.

    Matches the heavy-atom definition used by ``mdtraj.compute_contacts`` so
    that the prefilter and the distance calculation agree.
    """
    out = []
    for residue in topology.residues:
        idx = [
            atom.index
            for atom in residue.atoms
            if atom.element != md_element.hydrogen
        ]
        out.append(np.asarray(idx, dtype=np.int64))
    return out


def describe_residues(topology) -> list:
    """Build :class:`ResidueInfo` records for every residue in ``topology``."""
    infos = []
    for residue in topology.residues:
        chain = getattr(residue.chain, "chain_id", None)
        if not chain:
            chain = str(residue.chain.index)
        infos.append(
            ResidueInfo(
                index=residue.index,
                chain=str(chain),
                res_seq=int(residue.resSeq),
                name=str(residue.name),
                segment=str(getattr(residue, "segment_id", "") or ""),
            )
        )
    return infos


def make_pairs(n_residues: int, min_separation: int, skippable=None) -> np.ndarray:
    """All residue pairs (i, j) with ``j - i >= min_separation``.

    ``skippable`` is an optional boolean mask of residues to leave out entirely
    (used for residues that contain no heavy atoms after selection).
    """
    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")
    if n_residues <= min_separation:
        return np.zeros((0, 2), dtype=np.int64)

    i, j = np.triu_indices(n_residues, k=min_separation)
    pairs = np.column_stack([i, j]).astype(np.int64)
    if skippable is not None and np.any(skippable):
        keep = ~(skippable[pairs[:, 0]] | skippable[pairs[:, 1]])
        pairs = pairs[keep]
    return pairs


def _check_positive(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value


def continuous_sigma(cutoff_nm, d_max_nm=0.8, k_at_d_max=1e-5):
    """Gaussian width that makes the semi-Gaussian kernel equal ``k_at_d_max``
    at ``d_max_nm``.

    ``s = sqrt((c^2 - d_max^2) / (2 ln k))``, evaluated with the exact
    expression of trajcontacts 1.0.1. The defaults give 0.1378418789 nm for
    ``cutoff_nm = 0.45``.
    """
    cut = _check_positive("cutoff_nm", cutoff_nm)
    dcut = _check_positive("d_max_nm", d_max_nm)
    kdcut = _check_positive("k_at_d_max", k_at_d_max)
    if dcut <= cut:
        raise ValueError(
            f"d_max_nm ({dcut!r}) must be larger than cutoff_nm ({cut!r})"
        )
    if kdcut >= 1.0:
        raise ValueError(f"k_at_d_max must be between 0 and 1, got {kdcut!r}")
    return float(np.sqrt(((cut**2) - (dcut**2)) / (2 * np.log(kdcut))))


def kernel_support(cutoff_nm, sigma_nm, tol=1e-12):
    """Distance beyond which the semi-Gaussian weight is below ``tol``.

    Solves ``exp(-(d^2 - c^2) / 2s^2) = tol``, i.e.
    ``d = sqrt(c^2 - 2 s^2 ln tol)``.
    """
    cut = _check_positive("cutoff_nm", cutoff_nm)
    sigma = _check_positive("sigma_nm", sigma_nm)
    tol = _check_positive("tol", tol)
    if tol >= 1.0:
        raise ValueError(f"tol must be between 0 and 1, got {tol!r}")
    return float(np.sqrt(cut * cut - 2.0 * sigma * sigma * np.log(tol)))


def semi_gaussian_kernel(d, cutoff_nm, sigma_nm, legacy_rounding=False):
    """Semi-Gaussian contact weight of distances ``d`` (nm).

    1 for ``d <= cutoff_nm``, ``exp(-(d^2 - c^2) / (2 s^2))`` beyond.

    The default evaluates that form in float64. The comparison with the
    cutoff is made in the dtype of ``d`` -- exactly the comparison the binary
    contact count uses -- so a frame counted as a binary contact always has
    weight 1.

    ``legacy_rounding=True`` evaluates the expression of trajcontacts 1.0.1
    ``calc_switchFunctionK`` verbatim, with the same operand types (a Python
    float cutoff, a NumPy float64 sigma), including ``np.around(K, 4)``. For
    float32 distances under NumPy 2 this squares ``d`` in float32 and
    promotes to float64 at the division by the float64 sigma term, as 1.0.1
    does.
    """
    if legacy_rounding:
        dij = np.asarray(d)
        cut = float(cutoff_nm)
        sigma = np.float64(sigma_nm)
        K = np.where(
            dij <= cut,
            1,
            np.exp(-((dij**2) / (2 * sigma**2)))
            / np.exp(-((cut**2) / (2 * sigma**2))),
        )
        return np.around(K, 4)

    d = np.asarray(d)
    cut = float(cutoff_nm)
    sigma = float(sigma_nm)
    inside = d <= cut
    d64 = d.astype(np.float64, copy=False)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-(d64 * d64 - cut * cut) / (2.0 * sigma * sigma))
    return np.where(inside, 1.0, np.minimum(weights, 1.0))


def _residue_centroids_and_radii(traj, heavy):
    """Heavy-atom centroid and bounding radius of each residue, per frame.

    Returns ``(centroids, radii)`` with shapes (n_frames, n_res, 3) and
    (n_frames, n_res).  Residues with no heavy atoms get a NaN centroid and an
    infinite radius, which makes the prefilter keep any pair involving them.
    """
    n_frames = traj.n_frames
    n_res = len(heavy)
    centroids = np.full((n_frames, n_res, 3), np.nan, dtype=np.float32)
    radii = np.full((n_frames, n_res), np.inf, dtype=np.float32)

    xyz = traj.xyz
    for r, idx in enumerate(heavy):
        if idx.size == 0:
            continue
        coords = xyz[:, idx, :]                      # (n_frames, n_heavy, 3)
        centre = coords.mean(axis=1)                 # (n_frames, 3)
        centroids[:, r, :] = centre
        offsets = coords - centre[:, None, :]
        radii[:, r] = np.sqrt((offsets ** 2).sum(axis=-1)).max(axis=1)
    return centroids, radii


def _centroid_distances(centroids, unitcell_vectors, pairs, periodic):
    """Minimum-image distances between residue centroids for ``pairs``."""
    pseudo = md.Trajectory(xyz=centroids, topology=None)
    if periodic and unitcell_vectors is not None:
        pseudo.unitcell_vectors = unitcell_vectors
    use_pbc = periodic and pseudo.unitcell_vectors is not None
    return md.compute_distances(pseudo, pairs, periodic=use_pbc, opt=True)


def _prefilter_chunk(traj, heavy, pairs, cutoff, periodic, pair_block):
    """Indices into ``pairs`` that might be in contact somewhere in ``traj``."""
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64)

    centroids, radii = _residue_centroids_and_radii(traj, heavy)
    unitcell = traj.unitcell_vectors

    keep = np.zeros(len(pairs), dtype=bool)
    for start in range(0, len(pairs), pair_block):
        stop = min(start + pair_block, len(pairs))
        block = pairs[start:stop]
        dist = _centroid_distances(centroids, unitcell, block, periodic)
        slack = radii[:, block[:, 0]] + radii[:, block[:, 1]]
        with np.errstate(invalid="ignore"):
            lower_bound = dist - slack
            # NaN (empty residue) propagates as "cannot rule out".
            possible = ~(lower_bound > cutoff)
        keep[start:stop] = possible.any(axis=0)
    return np.flatnonzero(keep)


def _heavy_layout(heavy):
    """Flatten per-residue heavy-atom lists into ``(flat, offsets, lengths)``."""
    lengths = np.asarray([idx.size for idx in heavy], dtype=np.int64)
    offsets = np.zeros(len(heavy) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if len(heavy):
        flat = np.concatenate([np.asarray(idx, dtype=np.int64) for idx in heavy])
    else:
        flat = np.zeros(0, dtype=np.int64)
    return flat, offsets, lengths


def _closest_heavy_distances(traj, layout, block, periodic):
    """Minimum heavy-atom distance for each residue pair in ``block``.

    Numerically identical to ``mdtraj.compute_contacts(scheme="closest-heavy",
    ignore_nonprotein=False)``: the same atom pairs go through the same
    ``mdtraj.compute_distances`` call, and a minimum involves no rounding.
    What differs is the bookkeeping. MDTraj builds the atom-pair list with a
    Python ``itertools.product`` loop and locates each residue pair's slice
    with a prefix sum that is recomputed for every pair, which is quadratic
    in the block size; here both steps are vectorised and the reduction is a
    single ``np.minimum.reduceat``.

    Blocks that involve a residue without heavy atoms are handed to MDTraj
    unchanged, so the error behaviour for such input is also the same.
    """
    flat, offsets, lengths = layout
    first, second = block[:, 0], block[:, 1]
    n_first, n_second = lengths[first], lengths[second]
    if block.size == 0 or np.any(n_first == 0) or np.any(n_second == 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist, _ = md.compute_contacts(
                traj,
                contacts=block,
                scheme="closest-heavy",
                periodic=periodic,
                ignore_nonprotein=False,
            )
        return dist

    per_pair = n_first * n_second
    starts = np.zeros(len(block), dtype=np.int64)
    np.cumsum(per_pair[:-1], out=starts[1:])
    owner = np.repeat(np.arange(len(block), dtype=np.int64), per_pair)
    local = np.arange(int(per_pair.sum()), dtype=np.int64) - starts[owner]
    width = n_second[owner]
    # Same order as itertools.product(atoms_i, atoms_j) in MDTraj.
    atom_pairs = np.empty((local.size, 2), dtype=np.int32)
    atom_pairs[:, 0] = flat[offsets[first[owner]] + local // width]
    atom_pairs[:, 1] = flat[offsets[second[owner]] + local % width]
    del owner, local, width

    atom_dist = md.compute_distances(traj, atom_pairs, periodic=periodic)
    return np.minimum.reduceat(atom_dist, starts, axis=1)


@dataclass(frozen=True)
class _Reduction:
    """What one distance pass should produce.

    ``cutoff`` is set when binary frame counts are wanted, ``kernel`` (a
    ``(cutoff, sigma, legacy_rounding)`` tuple) when continuous sums are
    wanted, and ``radius`` is the distance the exact prefilter must keep.
    """

    cutoff: object = None
    kernel: object = None
    radius: float = 0.0


def _reduce_chunk(traj, heavy, pairs, spec, periodic, prefilter, pair_block):
    """Binary counts and/or continuous kernel sums over the frames in ``traj``.

    Returns ``(pair_indices, counts, sums, n_candidates)``. ``counts`` is
    ``None`` unless ``spec.cutoff`` is set and ``sums`` is ``None`` unless
    ``spec.kernel`` is set; only pairs with a non-zero entry are returned.
    """
    want_counts = spec.cutoff is not None
    want_sums = spec.kernel is not None
    empty_idx = np.zeros(0, dtype=np.int64)
    empty_counts = np.zeros(0, dtype=np.int64) if want_counts else None
    empty_sums = np.zeros(0, dtype=np.float64) if want_sums else None

    if len(pairs) == 0 or traj.n_frames == 0:
        return empty_idx, empty_counts, empty_sums, 0

    if prefilter:
        candidate_idx = _prefilter_chunk(
            traj, heavy, pairs, spec.radius, periodic, pair_block
        )
    else:
        candidate_idx = np.arange(len(pairs), dtype=np.int64)

    if candidate_idx.size == 0:
        return empty_idx, empty_counts, empty_sums, 0

    layout = _heavy_layout(heavy)
    counts = np.zeros(candidate_idx.size, dtype=np.int64) if want_counts else None
    sums = np.zeros(candidate_idx.size, dtype=np.float64) if want_sums else None
    for start in range(0, candidate_idx.size, pair_block):
        stop = min(start + pair_block, candidate_idx.size)
        block = pairs[candidate_idx[start:stop]]
        dist = _closest_heavy_distances(traj, layout, block, periodic)
        if want_counts:
            counts[start:stop] = (dist <= spec.cutoff).sum(axis=0)
        if want_sums:
            cutoff, sigma, legacy = spec.kernel
            weights = semi_gaussian_kernel(
                dist, cutoff, sigma, legacy_rounding=legacy
            )
            sums[start:stop] = weights.sum(axis=0)

    nonzero = np.zeros(candidate_idx.size, dtype=bool)
    if want_counts:
        nonzero |= counts > 0
    if want_sums:
        nonzero |= sums > 0
    return (
        candidate_idx[nonzero],
        counts[nonzero] if want_counts else None,
        sums[nonzero] if want_sums else None,
        int(candidate_idx.size),
    )


def _count_chunk(traj, heavy, pairs, cutoff, periodic, prefilter, pair_block):
    """Frames-in-contact for every pair, restricted to the frames in ``traj``.

    Returns ``(pair_indices, counts, n_candidates)``.
    """
    spec = _Reduction(cutoff=cutoff, radius=cutoff)
    idx, counts, _sums, n_candidates = _reduce_chunk(
        traj, heavy, pairs, spec, periodic, prefilter, pair_block
    )
    return idx, counts, n_candidates


# ---------------------------------------------------------------------------
# Parallel execution
#
# The 0.1.x code passed the Trajectory as a starmap argument, so the whole
# coordinate array was pickled once per task.  Here the trajectory is placed in
# a module global *before* the pool forks, so workers inherit it copy-on-write
# and only small index arrays cross the process boundary.
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _worker(bounds):
    first, last = bounds
    state = _WORKER_STATE
    sub = state["traj"][first:last]
    return _reduce_chunk(
        sub,
        state["heavy"],
        state["pairs"],
        state["spec"],
        state["periodic"],
        state["prefilter"],
        state["pair_block"],
    )


def _frame_chunk_bounds(n_frames, chunk):
    return [(s, min(s + chunk, n_frames)) for s in range(0, n_frames, chunk)]


def _choose_chunk_sizes(n_frames, n_pairs, memory_budget, n_processes=1):
    """Pick frame-chunk and pair-block sizes that respect a memory budget.

    ``memory_budget`` is a TOTAL scratch budget, not a per-chunk one. When a
    pool of ``n_processes`` workers is used, up to that many chunks can be
    in flight at once, so the per-chunk budget is ``memory_budget /
    n_processes``: worst case, all workers are busy simultaneously.
    Without this division, ``-n`` had no effect on the memory bound at all
    -- a run with many more chunks than a single chunk's budget assumed
    could use ``n_processes`` times the intended memory (observed in
    practice: a 78-chunk run with ``-n 128`` used ~150GB against a 2GB
    per-chunk target).
    """
    n_pairs = max(int(n_pairs), 1)
    n_processes = max(1, int(n_processes))
    # Scratch is dominated by (frame_chunk x pair_block) float64 arrays; a few
    # of those exist at once, hence the divisor.
    per_frame_pair = 8.0 * 6.0
    pair_block = min(n_pairs, 500_000)
    per_worker_budget = memory_budget / n_processes
    frame_chunk = int(per_worker_budget / (per_frame_pair * pair_block))
    frame_chunk = max(1, min(frame_chunk, n_frames if n_frames else 1))
    return frame_chunk, pair_block


def _run_reduction(
    traj, pairs, spec, periodic, n_processes, prefilter, memory_budget, progress
):
    """Drive :func:`_reduce_chunk` over frame chunks, serially or in a pool.

    Returns ``(pairs, counts, sums, n_candidates, periodic, residues)`` where
    ``counts``/``sums`` are full-length arrays (or ``None`` when not wanted).
    Chunks are accumulated in frame order, also in the pool, so continuous
    sums are reproducible run to run.
    """
    pairs = np.ascontiguousarray(pairs, dtype=np.int64).reshape(-1, 2)
    heavy = heavy_atom_indices(traj.topology)
    residues = describe_residues(traj.topology)

    if periodic and traj.unitcell_vectors is None:
        warnings.warn(
            "no unit cell information in the trajectory; "
            "distances will be computed without periodic boundary conditions",
            RuntimeWarning,
            stacklevel=3,
        )
        periodic = False

    frame_chunk, pair_block = _choose_chunk_sizes(
        traj.n_frames, len(pairs), memory_budget, n_processes=n_processes
    )
    bounds = _frame_chunk_bounds(traj.n_frames, frame_chunk)

    counts = np.zeros(len(pairs), dtype=np.int64) if spec.cutoff is not None else None
    sums = np.zeros(len(pairs), dtype=np.float64) if spec.kernel is not None else None
    candidates_seen = 0

    can_fork = "fork" in mp.get_all_start_methods()
    use_pool = n_processes > 1 and len(bounds) > 1 and can_fork
    if n_processes > 1 and not can_fork:
        warnings.warn(
            "this platform does not support the 'fork' start method; "
            "running serially instead",
            RuntimeWarning,
            stacklevel=3,
        )

    def _accumulate(result):
        nonlocal candidates_seen
        idx, chunk_counts, chunk_sums, n_candidates = result
        if idx.size:
            if counts is not None:
                counts[idx] += chunk_counts
            if sums is not None:
                sums[idx] += chunk_sums
        candidates_seen = max(candidates_seen, n_candidates)

    if use_pool:
        _WORKER_STATE.update(
            traj=traj,
            heavy=heavy,
            pairs=pairs,
            spec=spec,
            periodic=periodic,
            prefilter=prefilter,
            pair_block=pair_block,
        )
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_processes) as pool:
                for done, result in enumerate(pool.imap(_worker, bounds), start=1):
                    _accumulate(result)
                    if progress is not None:
                        progress(done, len(bounds))
        finally:
            _WORKER_STATE.clear()
    else:
        for done, (first, last) in enumerate(bounds, start=1):
            _accumulate(
                _reduce_chunk(
                    traj[first:last],
                    heavy,
                    pairs,
                    spec,
                    periodic,
                    prefilter,
                    pair_block,
                )
            )
            if progress is not None:
                progress(done, len(bounds))

    return pairs, counts, sums, candidates_seen, periodic, residues


def compute_contact_counts(
    traj,
    pairs,
    cutoff_nm,
    periodic=True,
    n_processes=1,
    prefilter=True,
    memory_budget=_DEFAULT_MEMORY_BUDGET,
    progress=None,
):
    """Count, for each pair, the frames whose minimum heavy-atom distance <= cutoff.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Already sliced to the subsystem of interest.
    pairs : ndarray, shape (n_pairs, 2)
        Residue index pairs, as produced by :func:`make_pairs`.
    cutoff_nm : float
        Contact cutoff in nanometres.
    periodic : bool
        Apply the minimum-image convention when the trajectory has a unit cell.
    n_processes : int
        Worker processes.  Only used when the platform supports ``fork``.
        ``memory_budget`` is divided across these, since that many chunks
        can be processed concurrently.
    prefilter : bool
        Enable the exact geometric prefilter.
    progress : callable or None
        Called as ``progress(chunks_done, chunks_total)`` after each chunk.

    Returns
    -------
    ContactResult
    """
    spec = _Reduction(cutoff=cutoff_nm, radius=cutoff_nm)
    pairs, counts, _sums, n_candidates, _periodic, residues = _run_reduction(
        traj, pairs, spec, periodic, n_processes, prefilter, memory_budget, progress
    )
    return ContactResult(
        pairs=pairs,
        counts=counts,
        n_frames=traj.n_frames,
        residues=residues,
        cutoff_nm=cutoff_nm,
        n_pairs_considered=len(pairs),
        n_pairs_prefiltered=max(0, len(pairs) - n_candidates),
    )


def _continuous_setup(
    cutoff_nm, sigma_nm, d_max_nm, k_at_d_max, legacy_rounding, support_tol,
    prefilter,
):
    """Resolve sigma, the kernel support and the prefilter radius."""
    _check_positive("cutoff_nm", cutoff_nm)
    if sigma_nm is None:
        sigma_nm = continuous_sigma(cutoff_nm, d_max_nm, k_at_d_max)
    _check_positive("sigma_nm", sigma_nm)
    sigma_nm = float(sigma_nm)
    legacy_rounding = bool(legacy_rounding)
    if legacy_rounding:
        # Beyond this distance np.around(K, 4) is exactly 0, so the support is
        # exact whether or not the prefilter runs.
        support = kernel_support(cutoff_nm, sigma_nm, _LEGACY_ZERO_LEVEL)
        radius = support + _SUPPORT_MARGIN_NM
    elif prefilter:
        support = kernel_support(cutoff_nm, sigma_nm, support_tol)
        radius = support + _SUPPORT_MARGIN_NM
    else:
        # No prefilter: every pair is evaluated, the kernel has full range.
        kernel_support(cutoff_nm, sigma_nm, support_tol)   # validates tol
        support = radius = float("inf")
    kernel = (float(cutoff_nm), sigma_nm, legacy_rounding)
    return kernel, sigma_nm, support, radius


def compute_continuous_contacts(
    traj,
    pairs,
    cutoff_nm,
    sigma_nm=None,
    d_max_nm=0.8,
    k_at_d_max=1e-5,
    legacy_rounding=False,
    support_tol=1e-12,
    periodic=True,
    n_processes=1,
    prefilter=True,
    memory_budget=_DEFAULT_MEMORY_BUDGET,
    progress=None,
):
    """Frame-averaged semi-Gaussian contact weight of every residue pair.

    Per frame, ``d`` is the minimum heavy-atom distance of a pair (as for
    :func:`compute_contact_counts`) and the weight is
    :func:`semi_gaussian_kernel` of ``d``: 1 within ``cutoff_nm``, decaying as
    a Gaussian beyond it. The weights are summed over frames; the mean is the
    continuous analogue of the binary contact fraction.

    Parameters
    ----------
    cutoff_nm : float
        Distance (nm) up to which the weight is exactly 1.
    sigma_nm : float or None
        Gaussian width (nm). ``None`` derives it from ``d_max_nm`` and
        ``k_at_d_max`` with :func:`continuous_sigma`.
    d_max_nm, k_at_d_max : float
        The weight equals ``k_at_d_max`` at ``d_max_nm``. Ignored when
        ``sigma_nm`` is given.
    legacy_rounding : bool
        Evaluate the kernel with the exact expression of trajcontacts 1.0.1,
        including rounding every per-frame weight to 4 decimals.
    support_tol : float
        Without legacy rounding, the prefilter drops a pair only when its
        per-frame weight is below ``support_tol`` in every frame of a chunk.
        With ``prefilter=False`` every pair is evaluated at every distance.
    periodic, n_processes, prefilter, memory_budget, progress
        As for :func:`compute_contact_counts`.

    Returns
    -------
    ContinuousContactResult
    """
    kernel, sigma_nm, support, radius = _continuous_setup(
        cutoff_nm, sigma_nm, d_max_nm, k_at_d_max, legacy_rounding,
        support_tol, prefilter,
    )
    spec = _Reduction(kernel=kernel, radius=radius)
    pairs, _counts, sums, n_candidates, _periodic, residues = _run_reduction(
        traj, pairs, spec, periodic, n_processes, prefilter, memory_budget, progress
    )
    return ContinuousContactResult(
        pairs=pairs,
        sums=sums,
        n_frames=traj.n_frames,
        residues=residues,
        cutoff_nm=float(cutoff_nm),
        sigma_nm=sigma_nm,
        support_nm=support,
        legacy_rounding=kernel[2],
        n_pairs_considered=len(pairs),
        n_pairs_prefiltered=max(0, len(pairs) - n_candidates),
    )


def compute_contacts_both(
    traj,
    pairs,
    cutoff_nm,
    sigma_nm=None,
    d_max_nm=0.8,
    k_at_d_max=1e-5,
    legacy_rounding=False,
    support_tol=1e-12,
    periodic=True,
    n_processes=1,
    prefilter=True,
    memory_budget=_DEFAULT_MEMORY_BUDGET,
    progress=None,
):
    """Binary counts and continuous sums from a single distance pass.

    Takes the arguments of :func:`compute_continuous_contacts`; the binary
    counts use the same ``cutoff_nm``. Returns
    ``(ContactResult, ContinuousContactResult)``, identical to calling
    :func:`compute_contact_counts` and :func:`compute_continuous_contacts`
    separately, except that ``ContactResult.n_pairs_prefiltered`` reflects the
    (larger) prefilter radius of the continuous kernel.
    """
    kernel, sigma_nm, support, radius = _continuous_setup(
        cutoff_nm, sigma_nm, d_max_nm, k_at_d_max, legacy_rounding,
        support_tol, prefilter,
    )
    spec = _Reduction(cutoff=cutoff_nm, kernel=kernel, radius=max(radius, cutoff_nm))
    pairs, counts, sums, n_candidates, _periodic, residues = _run_reduction(
        traj, pairs, spec, periodic, n_processes, prefilter, memory_budget, progress
    )
    n_prefiltered = max(0, len(pairs) - n_candidates)
    binary = ContactResult(
        pairs=pairs,
        counts=counts,
        n_frames=traj.n_frames,
        residues=residues,
        cutoff_nm=cutoff_nm,
        n_pairs_considered=len(pairs),
        n_pairs_prefiltered=n_prefiltered,
    )
    continuous = ContinuousContactResult(
        pairs=pairs,
        sums=sums,
        n_frames=traj.n_frames,
        residues=residues,
        cutoff_nm=float(cutoff_nm),
        sigma_nm=sigma_nm,
        support_nm=support,
        legacy_rounding=kernel[2],
        n_pairs_considered=len(pairs),
        n_pairs_prefiltered=n_prefiltered,
    )
    return binary, continuous
