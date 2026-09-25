# Changelog

## 0.4.0

### Added
- Continuous ("semi-Gaussian") contacts, as in trajcontacts 1.0.1 `-m cont`
  and Westerlund et al. 2020: per frame K(d) = 1 for d <= c and
  exp(-(d^2 - c^2) / 2 sigma^2) beyond, with
  sigma = sqrt((c^2 - d_max^2) / (2 ln k)); averaged over frames.
  - CLI: `-m/--mode {binary,continuous,both}` (aliases `norm`, `cont`),
    `-d/--dmax` (8.0 A), `-k/--kval` (1e-5), `--sigma` (A, overrides -d/-k),
    `--legacy-rounding`, `-w/--cmatfract-cont`
    (`contactMatrixFraction_continuous.dat`), `--cont-precision` (2),
    `--condensed-out`. The default mode is `binary`, so existing command
    lines produce the same files as before.
  - Library: `compute_continuous_contacts`, `compute_contacts_both` (both
    results from one distance pass), `ContinuousContactResult` (`means`,
    `matrix()`, `condensed()`, `adjacency_matrix()`), `continuous_sigma`,
    `semi_gaussian_kernel`, `kernel_support`.
  - io: `write_continuous_matrix`, `write_condensed`; `write_npz` accepts a
    `continuous=` result and adds `cont_sums`, `cont_means`, `cont_cutoff_nm`,
    `cont_sigma_nm`, `cont_support_nm`, `cont_legacy_rounding`. Existing keys
    are unchanged.
  - `--legacy-rounding` reproduces 1.0.1's `contactMatrixFraction_continuous.dat`
    byte for byte; the default kernel matches the allopath
    `semi_Gaussian_kernel` (to float32 precision, which allopath uses for
    `exp`).
  - The exact prefilter uses the kernel's support (legacy: where the rounded
    weight becomes 0; otherwise where it drops below 1e-12).

### Changed
- Per-pair minimum heavy-atom distances are computed with one vectorised
  `mdtraj.compute_distances` call per pair block and `np.minimum.reduceat`,
  instead of `mdtraj.compute_contacts`, whose atom-pair bookkeeping is a
  Python loop with a prefix sum recomputed per pair (quadratic in the block
  size). Same atom pairs, same distance routine, bit-identical results;
  3-17x faster end to end on 150-600 residue test systems.
- Worker results are accumulated in frame-chunk order (`imap` instead of
  `imap_unordered`), so floating-point sums do not depend on scheduling.
- `scipy` added to the `test` extra (used to check the condensed layout).

### Notes
- trajcontacts 1.0.1 `-m both` with more than one trajectory writes
  corrupted binary matrices (its running binary matrix aliases the
  continuous one). trajcontacts2 `-m both` is not affected; its binary output
  equals 1.0.1 `-m norm`.

## 0.3.1

### Fixed
- `--memory` was applied per chunk, not as a total budget: with `n_processes`
  workers, up to that many chunks could run concurrently, each independently
  sized to the full budget. A 6-trajectory, 72000-frame, 299-residue run with
  `-n 128` produced 78 chunks -- all dispatched at once, at ~2GB each -- and
  was OOM-killed at ~150GB against a 25GB request. `_choose_chunk_sizes` now
  divides `memory_budget` by `n_processes`, so total concurrent scratch stays
  within the requested budget regardless of worker count.

## 0.3.0

Renamed the project `trajcontacts` -> `trajcontacts2` (package, import name and
console command). There is no compatibility shim for the old name.

### Added
- `-f/--trajectory` accepts multiple files (`-f run1.nc run2.nc run3.nc`),
  concatenated in the given order into one trajectory before the contact
  calculation, for replicate runs or restart segments sharing a topology.
- A single `-f` argument that is not itself a recognised trajectory file is
  read instead as a text list of trajectory paths, one per line (`#` comments
  allowed, relative paths resolve against the list file's directory), so many
  segments can be named without a long command line.

## 0.2.0

Same contact definition, same science, substantially faster and safer. On a
150-residue / 40-frame test system the runtime dropped from 17.4 s on four
cores to 0.8 s on one, with bit-identical output.

### Added
- `-s/--select`: restrict the calculation to a subsystem using MDTraj atom
  selection syntax. Defaults to `protein or nucleic`. Previously water, ions
  and lipids were all included, which made solvated trajectories unusable.
- `--min-separation`: skip residue pairs that are close in sequence. Was
  hardcoded to 1, so covalently bonded neighbours always counted as contacts.
- `--begin`, `--end`, `--stride`: analyse a subset of frames.
- `--no-periodic`: disable the minimum-image convention.
- `--min-fraction`: drop rarely observed pairs from the pair lists.
- `--npz`: write all results to a compressed binary archive.
- `--no-matrices`: skip the dense N x N text matrices.
- `--memory`: scratch-memory budget used to size internal chunks.
- `--version`, `--quiet`, and progress reporting.
- Pair lists now carry chain ID and the residue numbering from the input file,
  so output maps back onto the original structure. `--legacy-format` restores
  the 0.1.x columns.
- A test suite, checked against a brute-force transcription of the 0.1.x
  algorithm, and CI on Linux and macOS across Python 3.9-3.12.
- A `LICENSE` file for the BSD 2-clause license that was already declared.

### Changed
- Distances now come from `mdtraj.compute_contacts` (vectorised C) instead of
  one `compute_distances` call per residue pair.
- Frames are processed in chunks, so peak memory is bounded rather than
  proportional to trajectory length times residue count.
- An exact geometric prefilter discards residue pairs that cannot be in
  contact. It is a lower bound, never an approximation, so no contact is lost.
- Multiprocessing no longer pickles the trajectory once per task. Workers
  inherit it copy-on-write through `fork`; only index arrays cross process
  boundaries.
- `optparse` replaced with `argparse`. Missing `-p`/`-f` now produce a usage
  message rather than a file-not-found error from deep inside MDTraj.
- Installed as a package with a `console_scripts` entry point, so the shebang
  is generated for the interpreter used to install it.
- `contact_num_only.dat` is now configurable via `--numeric-out`.

### Fixed
- Residues with no heavy atoms no longer crash with
  `ValueError: atom_pairs must be ndim 2`.
- The pair-list header printed `res1_index` twice and omitted the `fraction`
  column that was actually written.
- `multiprocessing.Pool` was never closed or joined.
- `setup.py` pointed at an unrelated repository URL.

### Compatibility
Running with `-s all --min-separation 1 --legacy-format` reproduces 0.1.x
output exactly. This is checked in the test suite.

## 0.1.3
Initial released version.
