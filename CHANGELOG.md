# Changelog

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
