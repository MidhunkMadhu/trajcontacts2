"""Tests for the continuous ("semi-Gaussian") contact mode.

Two references anchor the implementation:

* :func:`_trajcontacts101_continuous` transcribes the ``-m cont`` code path
  of trajcontacts 1.0.1 (``pairs_dist``, ``calc_switchFunctionK``,
  ``cut_dist_cont``) and must match ``legacy_rounding=True`` to 1e-12.
* :func:`_allopath_semi_binary` transcribes ``semi_Gaussian_kernel`` and
  ``distance_matrix_semi_bin_loop`` of the Westerlund et al. 2020 allopath
  code and must match the default (unrounded) kernel with sigma = 0.138 nm
  and no periodic boundaries.
"""

import warnings

import numpy as np
import mdtraj as md
import pytest

from test_contacts import _make_traj
from trajcontacts2 import (
    ContinuousContactResult,
    compute_contact_counts,
    compute_contacts_both,
    compute_continuous_contacts,
    continuous_sigma,
    kernel_support,
    make_pairs,
    semi_gaussian_kernel,
)
from trajcontacts2.core import (
    _LEGACY_ZERO_LEVEL,
    _SUPPORT_MARGIN_NM,
    _closest_heavy_distances,
    _heavy_layout,
    heavy_atom_indices,
)


@pytest.fixture(scope="module")
def traj():
    return _make_traj(n_res=24, n_frames=12, seed=7)


def _small_box(traj, length=2.0):
    """Copy of ``traj`` in a box small enough for minimum-image to matter."""
    boxed = md.Trajectory(xyz=traj.xyz.copy(), topology=traj.topology)
    boxed.unitcell_lengths = np.tile([length] * 3, (traj.n_frames, 1))
    boxed.unitcell_angles = np.tile([90.0] * 3, (traj.n_frames, 1))
    return boxed


# --------------------------------------------------------------------------
# Reference 1: trajcontacts 1.0.1, -m cont (transcribed verbatim)
# --------------------------------------------------------------------------

def _pairs_dist(topo, traj, res1, res2):
    l = []
    for i in res1.atoms:
        if (i.element.name != 'hydrogen'):
            for j in res2.atoms:
                if (j.element.name != 'hydrogen'):
                    l.append([i.index, j.index])
    atom_pairs = np.array(l)
    all_pairdist = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)
    return all_pairdist


def _calc_switchFunctionK(dij, sigma, cut):
    K = np.where(dij <= cut, 1, np.exp(-((dij**2) / (2 * sigma**2))) / np.exp(-((cut**2) / (2 * sigma**2))))
    return np.around(K, 4)


def _cut_dist_cont(topo, traj, res1, res2, cut, dcut, Kdcut):
    all_pairdist = _pairs_dist(topo, traj, res1, res2)
    each_frame_min = np.min(all_pairdist, axis=1)
    sigma = np.sqrt(((cut**2) - (dcut**2)) / (2 * np.log(Kdcut)))
    continous_binary = _calc_switchFunctionK(each_frame_min, sigma, cut)
    f = np.sum(continous_binary)
    return f


def _trajcontacts101_continuous(traj, c_ang=4.5, d_ang=8.0, k=1e-5):
    """Dense matrix of 1.0.1 ``contactMatrixFraction_continuous.dat`` values
    (before the %.2f formatting), for n_apart = 1."""
    cut = c_ang / 10.0
    dcut = d_ang / 10.0
    Kdcut = k
    topo = traj.topology
    upto = topo.n_residues
    cmat = np.zeros([upto, upto])
    for i in range(0, upto - 1):
        res1 = topo.residue(i)
        for j in range(i + 1, upto):
            res2 = topo.residue(j)
            f = _cut_dist_cont(topo, traj, res1, res2, cut, dcut, Kdcut)
            cmat[i][j] = f
            cmat[j][i] = f
    return cmat / (traj.n_frames * 1.0)


# --------------------------------------------------------------------------
# Reference 2: Westerlund et al. 2020 allopath (transcribed verbatim)
# --------------------------------------------------------------------------

def _get_atom_pairs(inds1, inds2):
    atom_pairs = np.zeros((len(inds1) * len(inds2), 2))
    counter = 0
    for k in range(0, len(inds1)):
        for l in range(0, len(inds2)):
            atom_pairs[counter, 0] = inds1[k]
            atom_pairs[counter, 1] = inds2[l]
            counter += 1
    atom_pairs = atom_pairs[0:counter, ::]
    return atom_pairs


def _semi_Gaussian_kernel(distances, std_dev=0.138, cutoff=0.45):
    cutoff_value = np.exp(-cutoff**2 / (2 * std_dev**2))
    gaussians = np.exp(-distances**2 / (2 * std_dev**2)) / cutoff_value
    gaussians[distances < cutoff] = 1.0
    return gaussians


def _allopath_semi_binary(traj, std_dev=0.138, cutoff=0.45, promote=False):
    """``ContactMap.distance_matrix_semi_bin_loop`` over all residues.

    ``promote=True`` casts the float32 distances to float64 before the kernel,
    removing allopath's float32 evaluation of ``exp`` so that only the
    formula is compared.
    """
    sub = traj.atom_slice(traj.topology.select('protein and !(type H)'))
    n_residues = sub.n_residues
    atom_inds = [sub.topology.select("resid " + str(i)) for i in range(n_residues)]
    cmap = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            atom_pairs = _get_atom_pairs(atom_inds[i], atom_inds[j])
            distances = md.compute_distances(sub, atom_pairs, periodic=False)
            min_distances = np.min(distances, axis=1)
            if promote:
                min_distances = min_distances.astype(np.float64)
            semi_gaussians = _semi_Gaussian_kernel(
                min_distances, cutoff=cutoff, std_dev=std_dev
            )
            cmap[i, j] = np.mean(semi_gaussians, axis=0)
            cmap[j, i] = np.mean(semi_gaussians, axis=0)
    return cmap


# --------------------------------------------------------------------------
# Reference comparisons
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prefilter", [True, False])
def test_legacy_matches_trajcontacts101(traj, prefilter):
    expected = _trajcontacts101_continuous(traj)
    pairs = make_pairs(traj.topology.n_residues, 1)
    result = compute_continuous_contacts(
        traj, pairs, 0.45, d_max_nm=0.8, k_at_d_max=1e-5,
        legacy_rounding=True, prefilter=prefilter,
    )
    np.testing.assert_allclose(result.matrix(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("prefilter", [True, False])
def test_legacy_matches_trajcontacts101_small_box(traj, prefilter):
    """Same, with a 2 nm box so that minimum-image distances are exercised."""
    boxed = _small_box(traj)
    expected = _trajcontacts101_continuous(boxed)
    pairs = make_pairs(boxed.topology.n_residues, 1)
    result = compute_continuous_contacts(
        boxed, pairs, 0.45, legacy_rounding=True, prefilter=prefilter,
    )
    np.testing.assert_allclose(result.matrix(), expected, rtol=0, atol=1e-12)


def test_legacy_matches_trajcontacts101_other_parameters(traj):
    expected = _trajcontacts101_continuous(traj, c_ang=4.0, d_ang=7.0, k=1e-3)
    pairs = make_pairs(traj.topology.n_residues, 1)
    result = compute_continuous_contacts(
        traj, pairs, 0.40, d_max_nm=0.70, k_at_d_max=1e-3, legacy_rounding=True,
    )
    np.testing.assert_allclose(result.matrix(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("prefilter", [True, False])
def test_unrounded_matches_allopath_formula(traj, prefilter):
    """Formula identity: allopath with float64 distances vs the default kernel."""
    expected = _allopath_semi_binary(traj, promote=True)
    pairs = make_pairs(traj.topology.n_residues, 1)
    result = compute_continuous_contacts(
        traj, pairs, 0.45, sigma_nm=0.138, periodic=False, prefilter=prefilter,
    )
    np.testing.assert_allclose(result.matrix(), expected, rtol=1e-12, atol=1e-12)


def test_unrounded_matches_allopath_verbatim(traj):
    """allopath evaluates exp() in float32, so agreement is at float32 level."""
    expected = _allopath_semi_binary(traj)
    pairs = make_pairs(traj.topology.n_residues, 1)
    result = compute_continuous_contacts(
        traj, pairs, 0.45, sigma_nm=0.138, periodic=False,
    )
    np.testing.assert_allclose(result.matrix(), expected, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------

def test_sigma_value():
    sigma = continuous_sigma(0.45)
    assert sigma == pytest.approx(0.1378418789, abs=1e-10)
    cut, dcut, Kdcut = 4.5 / 10.0, 8.0 / 10.0, 1e-5
    assert sigma == float(np.sqrt(((cut**2) - (dcut**2)) / (2 * np.log(Kdcut))))


@pytest.mark.parametrize("legacy", [False, True])
def test_kernel_is_one_inside_cutoff(legacy):
    sigma = continuous_sigma(0.45)
    d = np.array([0.0, 0.1, 0.3, 0.449, 0.45], dtype=np.float32)
    k = semi_gaussian_kernel(d, 0.45, sigma, legacy_rounding=legacy)
    np.testing.assert_array_equal(k, np.ones(5))


def test_legacy_kernel_is_bitwise_trajcontacts101():
    """Element-wise identity with 1.0.1's calc_switchFunctionK on float32
    distances, including its dtype path (float32 d**2, float64 sigma) and
    the rounding to 4 decimals. Evaluating the same formula another way
    (float64 d**2, or a Python-float sigma, which keeps the division in
    float32 under NumPy 2) flips the 4th decimal of some of these values."""
    cut, dcut, Kdcut = 4.5 / 10.0, 8.0 / 10.0, 1e-5
    sigma = np.sqrt(((cut**2) - (dcut**2)) / (2 * np.log(Kdcut)))
    rng = np.random.default_rng(0)
    d = rng.uniform(0.3, 0.9, 200_000).astype(np.float32)
    expected = _calc_switchFunctionK(d, sigma, cut)
    got = semi_gaussian_kernel(d, 0.45, continuous_sigma(0.45), legacy_rounding=True)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_kernel_at_dmax_equals_k():
    for cutoff, d_max, k_val in [(0.45, 0.8, 1e-5), (0.40, 0.7, 1e-3), (0.5, 1.2, 1e-8)]:
        sigma = continuous_sigma(cutoff, d_max, k_val)
        value = semi_gaussian_kernel(np.array([d_max]), cutoff, sigma)[0]
        assert value == pytest.approx(k_val, rel=1e-12)


def test_kernel_decreases_beyond_cutoff():
    sigma = continuous_sigma(0.45)
    d = np.linspace(0.45, 1.5, 200)
    k = semi_gaussian_kernel(d, 0.45, sigma)
    assert np.all(np.diff(k) <= 0)
    assert np.all((k >= 0) & (k <= 1))


def test_unrounded_kernel_matches_legacy_expression_before_rounding():
    sigma = continuous_sigma(0.45)
    d = np.linspace(0.3, 1.0, 500)
    legacy = np.where(
        d <= 0.45, 1,
        np.exp(-((d**2) / (2 * sigma**2))) / np.exp(-((0.45**2) / (2 * sigma**2))),
    )
    np.testing.assert_allclose(semi_gaussian_kernel(d, 0.45, sigma), legacy, rtol=1e-12)


def test_kernel_support():
    sigma = continuous_sigma(0.45)
    for tol in (1e-4, 1e-8, 1e-12):
        support = kernel_support(0.45, sigma, tol)
        value = semi_gaussian_kernel(np.array([support]), 0.45, sigma)[0]
        assert value == pytest.approx(tol, rel=1e-9)
    legacy_support = kernel_support(0.45, sigma, _LEGACY_ZERO_LEVEL)
    assert legacy_support == pytest.approx(0.7608154, abs=1e-6)
    beyond = np.array([legacy_support + _SUPPORT_MARGIN_NM, 0.9, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(
        semi_gaussian_kernel(beyond, 0.45, sigma, legacy_rounding=True), 0.0
    )
    within = np.array([legacy_support - _SUPPORT_MARGIN_NM], dtype=np.float32)
    assert semi_gaussian_kernel(within, 0.45, sigma, legacy_rounding=True)[0] > 0


def test_kernel_parameter_validation():
    with pytest.raises(ValueError):
        continuous_sigma(0.45, d_max_nm=0.45)
    with pytest.raises(ValueError):
        continuous_sigma(0.45, d_max_nm=0.3)
    with pytest.raises(ValueError):
        continuous_sigma(0.45, k_at_d_max=0.0)
    with pytest.raises(ValueError):
        continuous_sigma(0.45, k_at_d_max=1.0)
    with pytest.raises(ValueError):
        continuous_sigma(-0.45)
    with pytest.raises(ValueError):
        kernel_support(0.45, 0.138, tol=2.0)
    with pytest.raises(ValueError):
        kernel_support(0.45, 0.0)


# --------------------------------------------------------------------------
# Invariance of the reduction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("legacy", [False, True])
def test_continuous_chunking_and_multiprocessing(traj, legacy):
    pairs = make_pairs(traj.topology.n_residues, 1)
    whole = compute_continuous_contacts(traj, pairs, 0.45, legacy_rounding=legacy)
    chunked = compute_continuous_contacts(
        traj, pairs, 0.45, legacy_rounding=legacy, memory_budget=50_000
    )
    parallel = compute_continuous_contacts(
        traj, pairs, 0.45, legacy_rounding=legacy, memory_budget=50_000,
        n_processes=2,
    )
    np.testing.assert_allclose(chunked.sums, whole.sums, rtol=1e-12, atol=0)
    np.testing.assert_allclose(parallel.sums, whole.sums, rtol=1e-12, atol=0)


@pytest.mark.parametrize("legacy", [False, True])
def test_continuous_prefilter_invariance(legacy):
    # Larger than the fixture, so that the ~1.1 nm unrounded support still
    # leaves pairs for the prefilter to discard.
    traj = _make_traj(n_res=60, n_frames=8, seed=11)
    pairs = make_pairs(traj.topology.n_residues, 1)
    reference = compute_continuous_contacts(
        traj, pairs, 0.45, legacy_rounding=legacy, prefilter=False
    )
    filtered = compute_continuous_contacts(
        traj, pairs, 0.45, legacy_rounding=legacy, prefilter=True,
        memory_budget=50_000,
    )
    assert filtered.n_pairs_prefiltered > 0
    # Legacy: dropped pairs are exactly 0. Unrounded: each dropped frame
    # contributes < support_tol (1e-12) to the mean.
    atol = 0.0 if legacy else 1e-12
    np.testing.assert_allclose(filtered.means, reference.means, rtol=1e-12, atol=atol)
    assert np.isinf(reference.support_nm) != legacy


def test_prefilter_keeps_pairs_beyond_cutoff(traj):
    """The prefilter must use the kernel support, not the binary cutoff."""
    pairs = make_pairs(traj.topology.n_residues, 1)
    binary, continuous = compute_contacts_both(traj, pairs, 0.45, legacy_rounding=True)
    beyond = (binary.counts == 0) & (continuous.sums > 0)
    assert beyond.any()


def test_means_bounded_and_dominate_binary(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    for legacy in (False, True):
        cont = compute_continuous_contacts(traj, pairs, 0.45, legacy_rounding=legacy)
        binary = compute_contact_counts(traj, pairs, 0.45)
        assert np.all(cont.means >= 0.0) and np.all(cont.means <= 1.0)
        assert np.all(cont.means >= binary.fractions)
        matrix = cont.matrix()
        assert np.array_equal(matrix, matrix.T)
        assert np.all(np.diag(matrix) == 0)


@pytest.mark.parametrize("legacy", [False, True])
def test_both_mode_equals_separate_modes(traj, legacy):
    pairs = make_pairs(traj.topology.n_residues, 1)
    binary, cont = compute_contacts_both(
        traj, pairs, 0.45, legacy_rounding=legacy, memory_budget=50_000
    )
    alone_binary = compute_contact_counts(traj, pairs, 0.45, memory_budget=50_000)
    alone_cont = compute_continuous_contacts(
        traj, pairs, 0.45, legacy_rounding=legacy, memory_budget=50_000
    )
    np.testing.assert_array_equal(binary.counts, alone_binary.counts)
    np.testing.assert_allclose(cont.sums, alone_cont.sums, rtol=1e-12, atol=0)
    assert cont.sigma_nm == alone_cont.sigma_nm
    assert cont.support_nm == alone_cont.support_nm


def test_condensed_round_trips_with_squareform(traj, tmp_path):
    from scipy.spatial.distance import squareform
    from trajcontacts2.io import write_condensed

    pairs = make_pairs(traj.topology.n_residues, 2)
    cont = compute_continuous_contacts(traj, pairs, 0.45)
    n = cont.n_residues
    condensed = cont.condensed()
    assert condensed.shape == (n * (n - 1) // 2,)
    np.testing.assert_array_equal(squareform(condensed), cont.matrix())
    np.testing.assert_array_equal(squareform(cont.matrix(), checks=False), condensed)

    path = tmp_path / "condensed.txt"
    write_condensed(path, cont)
    np.testing.assert_array_equal(squareform(np.loadtxt(path)), cont.matrix())


def test_adjacency_threshold(traj):
    pairs = make_pairs(traj.topology.n_residues, 1)
    cont = compute_continuous_contacts(traj, pairs, 0.45)
    adjacency = cont.adjacency_matrix(0.5)
    assert np.array_equal(adjacency, (cont.matrix() >= 0.5).astype(np.int64))
    assert isinstance(cont, ContinuousContactResult)


def test_fast_distances_match_mdtraj_bitwise():
    """The vectorised closest-heavy distances equal mdtraj.compute_contacts."""
    base = _make_traj(n_res=60, n_frames=4, seed=3)
    triclinic = md.Trajectory(xyz=base.xyz.copy(), topology=base.topology)
    triclinic.unitcell_lengths = np.tile([2.5, 2.7, 2.6], (base.n_frames, 1))
    triclinic.unitcell_angles = np.tile([75.0, 80.0, 85.0], (base.n_frames, 1))
    pairs = make_pairs(base.topology.n_residues, 1)
    layout = _heavy_layout(heavy_atom_indices(base.topology))
    for system in (base, _small_box(base, 2.0), triclinic):
        for periodic in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected, _ = md.compute_contacts(
                    system, contacts=pairs, scheme="closest-heavy",
                    periodic=periodic, ignore_nonprotein=False,
                )
            got = _closest_heavy_distances(system, layout, pairs, periodic)
            assert got.dtype == expected.dtype
            np.testing.assert_array_equal(got, expected)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cli_system(tmp_path, n_res=10, n_frames=4):
    traj = _make_traj(n_res=n_res, n_frames=n_frames)
    pdb = tmp_path / "system.pdb"
    traj.save_pdb(str(pdb))
    return pdb


def _cli_outputs(tmp_path):
    return [
        "-o", str(tmp_path / "contact.dat"), "-x", str(tmp_path / "pairs.dat"),
        "-y", str(tmp_path / "counts.dat"), "-z", str(tmp_path / "frac.dat"),
        "-w", str(tmp_path / "cont.dat"), "--numeric-out", "none",
    ]


@pytest.mark.parametrize("mode", ["cont", "continuous"])
def test_cli_continuous_mode(tmp_path, mode):
    from trajcontacts2.cli import main

    pdb = _cli_system(tmp_path)
    code = main([
        "-p", str(pdb), "-f", str(pdb), "-s", "all", "-q", "-m", mode,
        *_cli_outputs(tmp_path),
        "--condensed-out", str(tmp_path / "condensed.txt"),
        "--npz", str(tmp_path / "cont.npz"),
    ])
    assert code == 0
    assert (tmp_path / "cont.dat").exists()
    assert not (tmp_path / "contact.dat").exists()
    assert not (tmp_path / "pairs.dat").exists()

    matrix = np.loadtxt(tmp_path / "cont.dat")
    assert matrix.shape == (10, 10)
    archive = np.load(tmp_path / "cont.npz")
    for key in ("cont_sums", "cont_means", "cont_cutoff_nm", "cont_sigma_nm",
                "cont_support_nm", "cont_legacy_rounding"):
        assert key in archive.files
    assert "counts" not in archive.files
    assert float(archive["cont_sigma_nm"]) == pytest.approx(0.1378418789, abs=1e-9)
    assert not bool(archive["cont_legacy_rounding"])

    from scipy.spatial.distance import squareform
    full = squareform(np.loadtxt(tmp_path / "condensed.txt"))
    np.testing.assert_allclose(matrix, np.round(full, 2), atol=1e-12)


def test_cli_both_mode(tmp_path):
    from trajcontacts2.cli import main

    pdb = _cli_system(tmp_path)
    code = main([
        "-p", str(pdb), "-f", str(pdb), "-s", "all", "-q", "-m", "both",
        "--legacy-rounding", "--cont-precision", "4",
        *_cli_outputs(tmp_path), "--npz", str(tmp_path / "both.npz"),
    ])
    assert code == 0
    for name in ("contact.dat", "pairs.dat", "counts.dat", "frac.dat", "cont.dat"):
        assert (tmp_path / name).exists(), name

    archive = np.load(tmp_path / "both.npz")
    assert archive["counts"].shape == archive["cont_sums"].shape
    assert bool(archive["cont_legacy_rounding"])
    fractions = archive["counts"] / float(archive["n_frames"])
    assert np.all(archive["cont_means"] >= fractions)
    written = (tmp_path / "cont.dat").read_text().split()
    assert all(len(v.split(".")[1]) == 4 for v in written)


def test_cli_binary_mode_has_no_continuous_outputs(tmp_path):
    from trajcontacts2.cli import main

    pdb = _cli_system(tmp_path)
    code = main([
        "-p", str(pdb), "-f", str(pdb), "-s", "all", "-q",
        *_cli_outputs(tmp_path), "--npz", str(tmp_path / "bin.npz"),
    ])
    assert code == 0
    assert not (tmp_path / "cont.dat").exists()
    archive = np.load(tmp_path / "bin.npz")
    assert not any(key.startswith("cont_") for key in archive.files)


@pytest.mark.parametrize(
    "bad",
    [
        ["-d", "3.0"],
        ["-d", "4.5"],
        ["-k", "0"],
        ["-k", "1.5"],
        ["--sigma", "-1"],
        ["--cont-precision", "-1"],
    ],
)
def test_cli_rejects_bad_continuous_parameters(tmp_path, bad):
    from trajcontacts2.cli import main

    pdb = _cli_system(tmp_path)
    with pytest.raises(SystemExit):
        main(["-p", str(pdb), "-f", str(pdb), "-s", "all", "-q", "-m", "cont", *bad])


def test_cli_rejects_unknown_mode(tmp_path):
    from trajcontacts2.cli import main

    pdb = _cli_system(tmp_path)
    with pytest.raises(SystemExit):
        main(["-p", str(pdb), "-f", str(pdb), "-s", "all", "-q", "-m", "fuzzy"])
