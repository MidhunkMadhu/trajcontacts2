"""Core contact-calculation routines for trajcontacts.

The public entry point is :func:`compute_contact_counts`, which returns, for
every requested residue pair, the number of frames in which the minimum
inter-residue heavy-atom distance falls at or below a cutoff.

The definition of a contact is unchanged from trajcontacts 0.1.x: a pair of
residues is in contact in a given frame when the shortest distance between any
two of their non-hydrogen atoms is <= ``cutoff``.  What changed is how that
number is obtained:

* distances come from :func:`mdtraj.compute_contacts` (a vectorised C routine)
  instead of one ``compute_distances`` call per residue pair;
* frames are processed in chunks so peak memory stays bounded;
* a cheap, *exact* geometric prefilter discards pairs that cannot possibly be
  in contact within a chunk, which for a folded protein removes the large
  majority of the O(N^2) pair list.

The prefilter is a lower bound, never an approximation.  For residues i and j,

    min_heavy_distance(i, j) >= centroid_distance(i, j) - radius(i) - radius(j)

where ``radius`` is the distance from a residue's heavy-atom centroid to its
outermost heavy atom.  A pair is only discarded when that lower bound exceeds
the cutoff in every frame of the chunk, so no contact can be missed.
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
    "ResidueInfo",
    "compute_contact_counts",
    "describe_residues",
    "heavy_atom_indices",
    "make_pairs",
]

# Bytes of scratch space the chunking heuristics aim to stay under.
_DEFAULT_MEMORY_BUDGET = 2_000_000_000


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


def _count_chunk(traj, heavy, pairs, cutoff, periodic, prefilter, pair_block):
    """Frames-in-contact for every pair, restricted to the frames in ``traj``.

    Returns ``(pair_indices, counts, n_candidates)``.
    """
    if len(pairs) == 0 or traj.n_frames == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0

    if prefilter:
        candidate_idx = _prefilter_chunk(
            traj, heavy, pairs, cutoff, periodic, pair_block
        )
    else:
        candidate_idx = np.arange(len(pairs), dtype=np.int64)

    if candidate_idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0

    totals = np.zeros(candidate_idx.size, dtype=np.int64)
    for start in range(0, candidate_idx.size, pair_block):
        stop = min(start + pair_block, candidate_idx.size)
        block = pairs[candidate_idx[start:stop]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist, _ = md.compute_contacts(
                traj,
                contacts=block,
                scheme="closest-heavy",
                periodic=periodic,
                ignore_nonprotein=False,
            )
        totals[start:stop] = (dist <= cutoff).sum(axis=0)

    nonzero = totals > 0
    return candidate_idx[nonzero], totals[nonzero], int(candidate_idx.size)


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
    return _count_chunk(
        sub,
        state["heavy"],
        state["pairs"],
        state["cutoff"],
        state["periodic"],
        state["prefilter"],
        state["pair_block"],
    )


def _frame_chunk_bounds(n_frames, chunk):
    return [(s, min(s + chunk, n_frames)) for s in range(0, n_frames, chunk)]


def _choose_chunk_sizes(n_frames, n_pairs, memory_budget):
    """Pick frame-chunk and pair-block sizes that respect a memory budget."""
    n_pairs = max(int(n_pairs), 1)
    # Scratch is dominated by (frame_chunk x pair_block) float64 arrays; a few
    # of those exist at once, hence the divisor.
    per_frame_pair = 8.0 * 6.0
    pair_block = min(n_pairs, 500_000)
    frame_chunk = int(memory_budget / (per_frame_pair * pair_block))
    frame_chunk = max(1, min(frame_chunk, n_frames if n_frames else 1))
    return frame_chunk, pair_block


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
    prefilter : bool
        Enable the exact geometric prefilter.
    progress : callable or None
        Called as ``progress(chunks_done, chunks_total)`` after each chunk.

    Returns
    -------
    ContactResult
    """
    pairs = np.ascontiguousarray(pairs, dtype=np.int64).reshape(-1, 2)
    heavy = heavy_atom_indices(traj.topology)
    residues = describe_residues(traj.topology)

    if periodic and traj.unitcell_vectors is None:
        warnings.warn(
            "no unit cell information in the trajectory; "
            "distances will be computed without periodic boundary conditions",
            RuntimeWarning,
            stacklevel=2,
        )
        periodic = False

    frame_chunk, pair_block = _choose_chunk_sizes(
        traj.n_frames, len(pairs), memory_budget
    )
    bounds = _frame_chunk_bounds(traj.n_frames, frame_chunk)

    totals = np.zeros(len(pairs), dtype=np.int64)
    candidates_seen = 0

    can_fork = "fork" in mp.get_all_start_methods()
    use_pool = n_processes > 1 and len(bounds) > 1 and can_fork
    if n_processes > 1 and not can_fork:
        warnings.warn(
            "this platform does not support the 'fork' start method; "
            "running serially instead",
            RuntimeWarning,
            stacklevel=2,
        )

    def _accumulate(result):
        nonlocal candidates_seen
        idx, counts, n_candidates = result
        if idx.size:
            totals[idx] += counts
        candidates_seen = max(candidates_seen, n_candidates)

    if use_pool:
        _WORKER_STATE.update(
            traj=traj,
            heavy=heavy,
            pairs=pairs,
            cutoff=cutoff_nm,
            periodic=periodic,
            prefilter=prefilter,
            pair_block=pair_block,
        )
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_processes) as pool:
                for done, result in enumerate(
                    pool.imap_unordered(_worker, bounds), start=1
                ):
                    _accumulate(result)
                    if progress is not None:
                        progress(done, len(bounds))
        finally:
            _WORKER_STATE.clear()
    else:
        for done, (first, last) in enumerate(bounds, start=1):
            _accumulate(
                _count_chunk(
                    traj[first:last],
                    heavy,
                    pairs,
                    cutoff_nm,
                    periodic,
                    prefilter,
                    pair_block,
                )
            )
            if progress is not None:
                progress(done, len(bounds))

    return ContactResult(
        pairs=pairs,
        counts=totals,
        n_frames=traj.n_frames,
        residues=residues,
        cutoff_nm=cutoff_nm,
        n_pairs_considered=len(pairs),
        n_pairs_prefiltered=max(0, len(pairs) - candidates_seen),
    )
