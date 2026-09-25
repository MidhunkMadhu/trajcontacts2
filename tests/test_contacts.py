"""Tests for trajcontacts2.

The key test is :func:`test_matches_bruteforce_reference`, which checks the
optimised implementation against a transcription of the original 0.1.x
algorithm (one ``compute_distances`` call per residue pair, minimum over heavy
atom pairs, count frames under the cutoff). That reference is slow but
obviously correct, which is exactly what a reference should be.
"""

import numpy as np
import mdtraj as md
import pytest

from trajcontacts2.core import (
    _choose_chunk_sizes,
    compute_contact_counts,
    describe_residues,
    heavy_atom_indices,
    make_pairs,
)


# --------------------------------------------------------------------------
# Synthetic test system
# --------------------------------------------------------------------------

_ATOMS = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("HA", "H")]
_OFFSETS = {
    "N": (-0.12, 0.0, 0.0),
    "CA": (0.0, 0.0, 0.0),
    "C": (0.12, 0.0, 0.0),
    "O": (0.15, 0.10, 0.0),
    "CB": (0.0, 0.15, 0.05),
    "HA": (0.0, -0.10, 0.0),
}


def _make_traj(n_res=24, n_frames=5, seed=1, unitcell=True):
    """A small, chemically plausible pseudo-peptide with hydrogens."""
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_frames, n_res * len(_ATOMS), 3), dtype=np.float32)
    for frame in range(n_frames):
        point = np.zeros(3)
        for res in range(n_res):
            step = rng.normal(size=3)
            step /= np.linalg.norm(step)
            point = point + step * 0.38
            point -= 0.02 * point * np.linalg.norm(point)
            for a, (name, _) in enumerate(_ATOMS):
                idx = res * len(_ATOMS) + a
                xyz[frame, idx] = point + np.array(_OFFSETS[name]) + rng.normal(0, 0.005, 3)

    top = md.Topology()
    chain = top.add_chain()
    for res in range(n_res):
        residue = top.add_residue("ALA", chain, resSeq=res + 1)
        for name, symbol in _ATOMS:
            top.add_atom(name, md.element.get_by_symbol(symbol), residue)

    traj = md.Trajectory(xyz=xyz, topology=top)
    if unitcell:
        traj.unitcell_lengths = np.tile([10.0, 10.0, 10.0], (n_frames, 1))
        traj.unitcell_angles = np.tile([90.0, 90.0, 90.0], (n_frames, 1))
    return traj


@pytest.fixture(scope="module")
def traj():
    return _make_traj()


# --------------------------------------------------------------------------
# Reference implementation (the 0.1.x algorithm, written for clarity)
# --------------------------------------------------------------------------

def _bruteforce_counts(traj, pairs, cutoff, periodic=True):
    top = traj.topology
    heavy = [
        [a.index for a in top.residue(r).atoms if a.element.name != "hydrogen"]
        for r in range(top.n_residues)
    ]
    counts = np.zeros(len(pairs), dtype=np.int64)
    for k, (i, j) in enumerate(pairs):
        atom_pairs = np.array([[a, b] for a in heavy[i] for b in heavy[j]])
        dist = md.compute_distances(traj, atom_pairs, periodic=periodic, opt=True)
        counts[k] = int((dist.min(axis=1) <= cutoff).sum())
    return counts


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_matches_bruteforce_reference(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    expected = _bruteforce_counts(traj, pairs, 0.45)
    result = compute_contact_counts(traj, pairs, 0.45, prefilter=False, n_processes=1)
    np.testing.assert_array_equal(result.counts, expected)


def test_prefilter_is_exact(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    reference = compute_contact_counts(traj, pairs, 0.45, prefilter=False)
    filtered = compute_contact_counts(traj, pairs, 0.45, prefilter=True)
    np.testing.assert_array_equal(reference.counts, filtered.counts)


def test_chunking_does_not_change_results(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    whole = compute_contact_counts(traj, pairs, 0.45)
    chunked = compute_contact_counts(traj, pairs, 0.45, memory_budget=50_000)
    np.testing.assert_array_equal(whole.counts, chunked.counts)


def test_multiprocessing_matches_serial(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    serial = compute_contact_counts(traj, pairs, 0.45, memory_budget=50_000)
    parallel = compute_contact_counts(
        traj, pairs, 0.45, n_processes=2, memory_budget=50_000
    )
    np.testing.assert_array_equal(serial.counts, parallel.counts)


def test_memory_budget_is_divided_across_processes():
    """Regression test for a real OOM: a 6-trajectory, 72000-frame, 299
    residue run with -n 128 produced 78 chunks, all dispatched at once since
    78 < 128. Each chunk was independently sized to the full memory_budget
    (2GB default), so actual peak usage was ~78x that -- ~150GB against a
    25GB SLURM request. memory_budget must bound TOTAL concurrent scratch,
    not each chunk's alone.
    """
    n_frames, n_pairs, n_processes = 72_000, 44_551, 128
    budget = 2_000_000_000
    per_frame_pair = 8.0 * 6.0

    frame_chunk_serial, pair_block = _choose_chunk_sizes(
        n_frames, n_pairs, budget, n_processes=1
    )
    frame_chunk_pool, pair_block_pool = _choose_chunk_sizes(
        n_frames, n_pairs, budget, n_processes=n_processes
    )
    assert pair_block_pool == pair_block

    n_chunks = -(-n_frames // frame_chunk_pool)
    concurrent = min(n_processes, n_chunks)
    total_scratch = concurrent * frame_chunk_pool * pair_block_pool * per_frame_pair
    assert total_scratch <= budget * 1.05, (
        f"{concurrent} concurrent chunks x {frame_chunk_pool} frames would "
        f"use {total_scratch / 1e9:.1f}GB against a {budget / 1e9:.1f}GB budget"
    )
    # dividing the budget must actually shrink chunks relative to n_processes=1
    assert frame_chunk_pool <= frame_chunk_serial


@pytest.mark.parametrize("cutoff", [0.35, 0.45, 0.60])
def test_cutoffs_are_monotonic(traj, cutoff):
    pairs = make_pairs(traj.topology.n_residues, 1)
    counts = compute_contact_counts(traj, pairs, cutoff).counts
    larger = compute_contact_counts(traj, pairs, cutoff + 0.1).counts
    assert np.all(larger >= counts)


def test_hydrogens_are_ignored(traj):
    """Deleting hydrogens must not change any contact count."""
    pairs = make_pairs(traj.topology.n_residues, 1)
    with_h = compute_contact_counts(traj, pairs, 0.45).counts
    heavy_only = traj.atom_slice(traj.topology.select("element != H"))
    without_h = compute_contact_counts(heavy_only, pairs, 0.45).counts
    np.testing.assert_array_equal(with_h, without_h)


def test_min_separation_excludes_sequence_neighbours():
    traj = _make_traj(n_res=12, n_frames=3)
    for separation in (1, 2, 3):
        pairs = make_pairs(traj.topology.n_residues, separation)
        assert np.all(pairs[:, 1] - pairs[:, 0] >= separation)
    assert len(make_pairs(12, 1)) > len(make_pairs(12, 3))


def test_residues_without_heavy_atoms_are_skipped():
    """A water-hydrogen-only residue must not crash the pair builder."""
    traj = _make_traj(n_res=6, n_frames=2)
    top = traj.topology
    chain = top.add_chain()
    residue = top.add_residue("DUM", chain, resSeq=999)
    top.add_atom("H1", md.element.hydrogen, residue)
    xyz = np.concatenate(
        [traj.xyz, np.zeros((traj.n_frames, 1, 3), dtype=np.float32)], axis=1
    )
    patched = md.Trajectory(xyz=xyz, topology=top)
    patched.unitcell_lengths = traj.unitcell_lengths
    patched.unitcell_angles = traj.unitcell_angles

    heavy = heavy_atom_indices(patched.topology)
    empty = np.array([h.size == 0 for h in heavy])
    assert empty.sum() == 1

    pairs = make_pairs(patched.topology.n_residues, 1, skippable=empty)
    bad = patched.topology.n_residues - 1
    assert not np.any(pairs == bad)
    result = compute_contact_counts(patched, pairs, 0.45)
    assert len(result.counts) == len(pairs)


def test_matrices_are_symmetric_and_thresholded(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    result = compute_contact_counts(traj, pairs, 0.45)

    counts = result.count_matrix()
    assert np.array_equal(counts, counts.T)
    assert np.all(np.diag(counts) == 0)

    fractions = result.fraction_matrix()
    assert fractions.max() <= 1.0 + 1e-9

    adjacency = result.adjacency_matrix(0.75)
    assert set(np.unique(adjacency)).issubset({0, 1})
    assert np.array_equal(adjacency, (fractions >= 0.75).astype(np.int64))


def test_residue_labels_preserve_input_numbering(traj):
    infos = describe_residues(traj.topology)
    assert [r.res_seq for r in infos] == list(range(1, traj.topology.n_residues + 1))
    assert all(r.name == "ALA" for r in infos)
    assert infos[0].label.endswith("ALA1")


def test_missing_unitcell_warns_and_falls_back():
    traj = _make_traj(n_res=8, n_frames=2, unitcell=False)
    pairs = make_pairs(traj.topology.n_residues, 1)
    with pytest.warns(RuntimeWarning, match="no unit cell"):
        result = compute_contact_counts(traj, pairs, 0.45, periodic=True)
    assert result.n_frames == 2


def test_cli_end_to_end(tmp_path):
    from trajcontacts2.cli import main

    traj = _make_traj(n_res=10, n_frames=4)
    pdb = tmp_path / "system.pdb"
    traj.save_pdb(str(pdb))

    out = tmp_path / "contact.dat"
    pairs_out = tmp_path / "pairs.dat"
    code = main([
        "-p", str(pdb), "-f", str(pdb), "-s", "all", "-q",
        "-o", str(out), "-x", str(pairs_out),
        "-y", str(tmp_path / "counts.dat"),
        "-z", str(tmp_path / "frac.dat"),
        "--numeric-out", "none",
        "--npz", str(tmp_path / "result.npz"),
    ])
    assert code == 0
    assert out.exists() and pairs_out.exists()
    assert np.loadtxt(out).shape == (10, 10)

    archive = np.load(tmp_path / "result.npz")
    assert archive["counts"].shape[0] == archive["pairs"].shape[0]
    assert int(archive["n_frames"]) == 4


def _save_segments(tmp_path, n_res=8, n_frames=6, split=3):
    """A reference topology plus two trajectory segments that together cover
    ``n_frames`` frames, for testing multi-file -f handling."""
    traj = _make_traj(n_res=n_res, n_frames=n_frames)
    topology_pdb = tmp_path / "system.pdb"
    traj[:1].save_pdb(str(topology_pdb))
    seg1 = tmp_path / "seg1.pdb"
    seg2 = tmp_path / "seg2.pdb"
    traj[:split].save_pdb(str(seg1))
    traj[split:].save_pdb(str(seg2))
    return topology_pdb, seg1, seg2, n_frames


def test_cli_accepts_multiple_trajectory_files(tmp_path):
    from trajcontacts2.cli import main

    topology_pdb, seg1, seg2, n_frames = _save_segments(tmp_path)
    code = main([
        "-p", str(topology_pdb), "-f", str(seg1), str(seg2), "-s", "all", "-q",
        "-o", str(tmp_path / "contact.dat"), "-x", str(tmp_path / "pairs.dat"),
        "-y", str(tmp_path / "counts.dat"), "-z", str(tmp_path / "frac.dat"),
        "--numeric-out", "none", "--npz", str(tmp_path / "multi.npz"),
    ])
    assert code == 0
    archive = np.load(tmp_path / "multi.npz")
    assert int(archive["n_frames"]) == n_frames


def test_cli_accepts_trajectory_list_file(tmp_path):
    from trajcontacts2.cli import main

    topology_pdb, seg1, seg2, n_frames = _save_segments(tmp_path)
    listfile = tmp_path / "segments.txt"
    listfile.write_text(f"# trajectory segments, one per line\n{seg1.name}\n{seg2.name}\n")

    code = main([
        "-p", str(topology_pdb), "-f", str(listfile), "-s", "all", "-q",
        "-o", str(tmp_path / "contact.dat"), "-x", str(tmp_path / "pairs.dat"),
        "-y", str(tmp_path / "counts.dat"), "-z", str(tmp_path / "frac.dat"),
        "--numeric-out", "none", "--npz", str(tmp_path / "list.npz"),
    ])
    assert code == 0
    archive = np.load(tmp_path / "list.npz")
    assert int(archive["n_frames"]) == n_frames


def test_list_file_with_missing_entry_errors(tmp_path):
    from trajcontacts2.cli import main

    topology_pdb, seg1, _seg2, _n_frames = _save_segments(tmp_path)
    listfile = tmp_path / "segments.txt"
    listfile.write_text(f"{seg1.name}\nno_such_file.xtc\n")

    with pytest.raises(SystemExit):
        main(["-p", str(topology_pdb), "-f", str(listfile), "-s", "all", "-q"])


def test_resolve_trajectory_files():
    from trajcontacts2.cli import build_parser, resolve_trajectory_files

    parser = build_parser()
    assert resolve_trajectory_files(["a.pdb", "b.xtc"], parser) == ["a.pdb", "b.xtc"]
    assert resolve_trajectory_files(["run.dcd"], parser) == "run.dcd"
