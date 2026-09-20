# trajcontacts2

Residue-residue contacts from molecular dynamics trajectories and PDB structures.

Two residues in a macromolecular system are taken to be in contact in a given
frame when the shortest distance between any two of their heavy (non-hydrogen)
atoms falls within a cutoff, typically 4-5 Å. Contacts that persist for a
majority of the simulation time (75% by default) define the contact network of
the structure. `trajcontacts2` extracts those contacts from long MD trajectories
as well as from single PDB structures, and writes them both as a per-pair list
and as adjacency matrices ready for network analysis.

Built on [MDTraj](https://www.mdtraj.org/), so any topology and trajectory
format MDTraj can read will work.

## Installation

```bash
pip install trajcontacts2
```

Or from source:

```bash
git clone https://github.com/MidhunkMadhu/trajcontacts2.git
cd trajcontacts2
pip install .
```

Requires Python 3.9 or newer.

## Quick start

A single PDB structure (pass the same file as topology and coordinates):

```bash
trajcontacts2 -p 3sn6.pdb -f 3sn6.pdb
```

A solvated trajectory, protein only, excluding sequence neighbours up to i+2:

```bash
trajcontacts2 -p system.prmtop -f prod.nc --min-separation 3 -n 8
```

Every tenth frame of a long trajectory, 4 Å cutoff, keeping only contacts
present in at least 75% of frames:

```bash
trajcontacts2 -p system.gro -f prod.xtc --stride 10 -c 4.0 -a 75 --min-fraction 0.75
```

`trajcontacts2 -h` lists every option.

## Multiple trajectories

`-f/--trajectory` accepts more than one file, which are concatenated (in the
order given) into a single trajectory before the contact calculation. This is
for replicate runs or restart segments that share one topology:

```bash
trajcontacts2 -p system.prmtop -f run1.nc run2.nc run3.nc
```

For many segments, listing them all on the command line is unwieldy, so a
single `-f` argument that is not itself a recognised trajectory file is read
instead as a plain-text list of trajectory paths, one per line (`#` starts a
comment, relative paths resolve against the list file's own directory):

```
# segments.txt
run1.nc
run2.nc
run3.nc
```

```bash
trajcontacts2 -p system.prmtop -f segments.txt
```

Passing a single actual trajectory file to `-f`, as in the examples above,
works exactly as before.

## Selecting the subsystem

By default only `protein or nucleic` atoms are analysed. This matters: a
solvated box contains tens of thousands of water residues, and including them
turns an O(N²) pair list into something that will not finish. Use
`-s/--select` with any
[MDTraj selection expression](https://www.mdtraj.org/latest/atom_selection.html):

```bash
trajcontacts2 -p top.psf -f traj.dcd -s "protein and chainid 0 1"
trajcontacts2 -p top.psf -f traj.dcd -s "protein or resname POPC"
trajcontacts2 -p top.psf -f traj.dcd -s all      # 0.1.x behaviour
```

Residue indices in the output are 0-based indices into the *selected*
subsystem. The chain ID and residue number from the input file are reported
alongside them so results map back onto the original structure.

## Output files

| File | Option | Contents |
| --- | --- | --- |
| `contactResNames.dat` | `-x` | One row per residue pair: indices, chain, residue number, residue name, frames in contact, fraction |
| `contact_num_only.dat` | `--numeric-out` | The same pairs, indices and counts only |
| `contactMatrixResFrames.dat` | `-y` | Symmetric matrix of frame counts |
| `contactMatrixFraction.dat` | `-z` | Weighted adjacency matrix (fraction of frames) |
| `contact.dat` | `-o` | Unweighted adjacency matrix, 1 where fraction ≥ `-a` |
| `result.npz` | `--npz` | All of the above in one compressed binary archive |

Pass `none` to `--numeric-out` to skip that file, or `--no-matrices` to skip
the three dense matrices, which for large subsystems dominate both runtime and
disk.

Loading the archive in Python:

```python
import numpy as np

data = np.load("result.npz")
pairs, fractions = data["pairs"], data["fractions"]
stable = pairs[fractions >= 0.75]
```

## Using it as a library

```python
import mdtraj as md
from trajcontacts2 import compute_contact_counts, make_pairs

traj = md.load("prod.xtc", top="system.gro")
traj = traj.atom_slice(traj.topology.select("protein"))

pairs = make_pairs(traj.topology.n_residues, min_separation=3)
result = compute_contact_counts(traj, pairs, cutoff_nm=0.45)

adjacency = result.adjacency_matrix(0.75)
```

To load several trajectory segments as a library user, pass a list of paths to
`mdtraj.load()` directly, the same way the CLI's `-f` does:

```python
traj = md.load(["run1.nc", "run2.nc", "run3.nc"], top="system.prmtop")
```

## Notes on the calculation

- **Periodic boundaries.** The minimum-image convention is applied whenever the
  trajectory carries a unit cell. Disable it with `--no-periodic`. If a single
  molecule spans more than half the box, make it whole before running.
- **Heavy atoms.** Any atom whose element is not hydrogen counts as heavy,
  which is also how MDTraj defines it. Virtual sites, extra points and Drude
  particles therefore count as heavy atoms if your selection includes them; the
  default selection does not.
- **Sequence neighbours.** `--min-separation 1` (the default) includes
  covalently bonded i/i+1 pairs, which are always in contact. Most contact-map
  analyses want 2 or 3.
- **Pair prefilter.** Pairs whose residue centroids are too far apart, allowing
  for each residue's bounding radius, are skipped. This is a strict lower bound
  on the heavy-atom distance, so it never discards a real contact. It can be
  turned off with `--no-prefilter`.

## Performance

Version 0.2.0 replaced a per-pair Python loop over `mdtraj.compute_distances`
with a chunked, prefiltered call to `mdtraj.compute_contacts`. On a
150-residue, 40-frame test system:

| | wall time | cores |
| --- | --- | --- |
| 0.1.3 | 17.4 s | 4 |
| 0.2.0 | 0.8 s | 1 |

Output is bit-identical. Because the distance kernel is now vectorised, `-n`
usually makes little difference; raise it for very large systems, where the
pair list rather than the kernel is the bottleneck.

Memory is bounded by `--memory` (2 GB of scratch by default) rather than by
trajectory length. This is a total budget shared across `-n` worker
processes, not a per-worker one, so raising `-n` does not raise peak memory.
For trajectories too large to load at all, use `--stride`, or split the
analysis across `-f` segments.

## Migrating from 0.1.x

All 0.1.x flags still work and keep their meaning. Three defaults changed:

- Only `protein or nucleic` is analysed. Pass `-s all` for the old behaviour.
- Pair-list files carry chain and residue-number columns. Pass
  `--legacy-format` for the old columns.
- The pair-list header was corrected: it previously printed `res1_index` twice
  and omitted the `fraction` column that was in fact written.

`trajcontacts2 -p x.pdb -f x.pdb -s all --min-separation 1 --legacy-format`
reproduces 0.1.x output exactly, and the test suite checks this.

The package and command were renamed from `trajcontacts` to `trajcontacts2`
in 0.3.0; there is no compatibility shim, so update scripts and installs to
the new name.

## Development

```bash
pip install -e ".[test]"
pytest
```

The test suite validates the optimised code against a brute-force transcription
of the original algorithm, and checks that the prefilter, frame chunking and
multiprocessing paths all give identical counts.

## Citation

If you use this program, please cite:

> Madhu, M. K., et al. "Delineating the Biased Signaling Mechanism in Mutated
> Variants of β2-Adrenergic Receptor Using Molecular Dynamics Simulations"

## License

BSD 2-Clause. See [LICENSE](LICENSE).

Copyright: Computational Biophysics and Soft Matter Group, IISER Bhopal
(https://home.iiserb.ac.in/~rkm/)
