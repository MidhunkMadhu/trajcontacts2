"""Command-line interface for trajcontacts."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import __version__

EPILOG = """\
examples:
  # contacts from a single PDB structure
  trajcontacts2 -p 3sn6.pdb -f 3sn6.pdb

  # a solvated trajectory, protein only, 4 A cutoff, skip i to i+2 neighbours
  trajcontacts2 -p system.prmtop -f prod.nc -c 4.0 --min-separation 3 -n 8

  # analyse every 10th frame of a long trajectory
  trajcontacts2 -p system.gro -f prod.xtc --stride 10

  # keep only contacts present in at least 75% of frames, as an edge list
  trajcontacts2 -p ref.pdb -f traj.dcd -a 75 --min-fraction 0.75

  # several trajectory segments/replicates, concatenated and analysed as one
  trajcontacts2 -p system.prmtop -f run1.nc run2.nc run3.nc

  # same, but the segments are named in a text file, one path per line
  trajcontacts2 -p system.prmtop -f segments.txt

  # continuous (semi-Gaussian) contact matrix instead of the binary one
  trajcontacts2 -p system.prmtop -f prod.nc -m continuous

  # binary and continuous from one pass, trajcontacts 1.0.1 compatible
  trajcontacts2 -p top.pdb -f traj.dcd -s all -m both --legacy-rounding

  # Westerlund et al. 2020 semi-binary map (sigma 1.38 A, no PBC)
  trajcontacts2 -p top.pdb -f traj.dcd -s protein -m cont --sigma 1.38 \\
      --no-periodic --condensed-out semi_bin.txt
"""

_MODE_ALIASES = {
    "binary": "binary",
    "norm": "binary",
    "continuous": "continuous",
    "cont": "continuous",
    "both": "both",
}


def _parse_mode(value):
    mode = _MODE_ALIASES.get(str(value).lower())
    if mode is None:
        raise argparse.ArgumentTypeError(
            f"invalid choice: {value!r} (choose from binary, continuous, both; "
            "'norm' and 'cont' are accepted as aliases)"
        )
    return mode


def build_parser():
    parser = argparse.ArgumentParser(
        prog="trajcontacts2",
        description=(
            "Calculate residue-residue contacts from MD trajectories or PDB "
            "structures. Two residues are in contact in a frame when the "
            "shortest distance between any two of their heavy atoms is within "
            "the cutoff."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"trajcontacts2 {__version__}")

    req = parser.add_argument_group("input (required)")
    req.add_argument(
        "-p", "--parmtop", dest="topology", required=True, metavar="FILE",
        help="topology/parameter file (.prmtop/.psf/.gro/.pdb or any format "
             "MDTraj can read)",
    )
    req.add_argument(
        "-f", "--trajectory", dest="trajectory", required=True, nargs="+",
        metavar="FILE [FILE ...]",
        help="coordinate/trajectory file(s) (.nc/.xtc/.dcd/.gro/.pdb or any "
             "format MDTraj can read). Pass the same file as -p to analyse a "
             "single PDB structure. Multiple files are concatenated into one "
             "trajectory, in the order given. A single argument that is not "
             "itself a recognised trajectory file is instead read as a text "
             "list of trajectory paths, one per line ('#' comments allowed).",
    )

    sel = parser.add_argument_group("system selection")
    sel.add_argument(
        "-s", "--select", dest="selection", default="protein or nucleic",
        metavar="EXPR",
        help="MDTraj atom-selection expression defining the subsystem "
             "(default: '%(default)s'). Use 'all' to reproduce the 0.1.x "
             "behaviour of including water, ions and lipids.",
    )
    sel.add_argument(
        "--min-separation", dest="min_separation", type=int, default=1,
        metavar="N",
        help="only consider residue pairs at least N positions apart in the "
             "residue list (default: %(default)s, i.e. include covalently "
             "bonded sequence neighbours). Use 2 or 3 to exclude trivial "
             "backbone-neighbour contacts.",
    )

    frames = parser.add_argument_group("frame selection")
    frames.add_argument(
        "--begin", type=int, default=0, metavar="N",
        help="first frame to analyse, 0-based (default: %(default)s)",
    )
    frames.add_argument(
        "--end", type=int, default=None, metavar="N",
        help="stop before this frame (default: end of trajectory)",
    )
    frames.add_argument(
        "--stride", type=int, default=1, metavar="N",
        help="analyse every Nth frame (default: %(default)s)",
    )

    calc = parser.add_argument_group("contact definition")
    calc.add_argument(
        "-c", "--cutoff", dest="cutoff", type=float, default=4.5, metavar="A",
        help="heavy-atom cutoff distance in angstrom (default: %(default)s)",
    )
    calc.add_argument(
        "-a", "--cutpercent", dest="cutpercent", type=float, default=75.0,
        metavar="PCT",
        help="percentage of frames a contact must be present in to appear in "
             "the unweighted adjacency matrix (default: %(default)s)",
    )
    calc.add_argument(
        "-m", "--mode", dest="mode", type=_parse_mode, default="binary",
        metavar="{binary,continuous,both}",
        help="contact definition: 'binary' (in contact or not, the default), "
             "'continuous' (semi-Gaussian weight, see below), or 'both' from a "
             "single distance pass. 'norm' and 'cont' are accepted as aliases, "
             "as in trajcontacts 1.0.1.",
    )
    calc.add_argument(
        "--no-periodic", dest="periodic", action="store_false",
        help="disable the minimum-image convention. Periodic distances are "
             "used by default when the trajectory carries a unit cell.",
    )

    cont = parser.add_argument_group(
        "continuous contacts (-m continuous/both)",
        "Per frame, the weight of a residue pair is 1 when its minimum\n"
        "heavy-atom distance d is <= the cutoff c, and\n"
        "exp(-(d^2 - c^2) / (2 sigma^2)) beyond it. The mean weight over\n"
        "frames is written as a weighted adjacency matrix.",
    )
    cont.add_argument(
        "-d", "--dmax", dest="dmax", type=float, default=8.0, metavar="A",
        help="distance in angstrom at which the weight has decayed to -k "
             "(default: %(default)s). Sets sigma = sqrt((c^2 - dmax^2) / "
             "(2 ln k)).",
    )
    cont.add_argument(
        "-k", "--kval", dest="kval", type=float, default=1e-5, metavar="K",
        help="weight at -d/--dmax, between 0 and 1 (default: %(default)s)",
    )
    cont.add_argument(
        "--sigma", dest="sigma", type=float, default=None, metavar="A",
        help="Gaussian width in angstrom; overrides -d/--dmax and -k/--kval "
             "(e.g. 1.38 for Westerlund et al. 2020). Default: derived from "
             "-d and -k (1.378 A with the default -c/-d/-k).",
    )
    cont.add_argument(
        "--legacy-rounding", dest="legacy_rounding", action="store_true",
        help="evaluate the weight exactly as trajcontacts 1.0.1 did, rounding "
             "each per-frame weight to 4 decimals",
    )
    cont.add_argument(
        "-w", "--cmatfract-cont", "--cmatfract_cont", dest="cont_matrix_out",
        default="contactMatrixFraction_continuous.dat", metavar="FILE",
        help="matrix of mean continuous weights (default: %(default)s). "
             "Skipped by --no-matrices; pass 'none' to skip it alone.",
    )
    cont.add_argument(
        "--cont-precision", dest="cont_precision", type=int, default=2,
        metavar="N",
        help="decimals written to the -w matrix (default: %(default)s, as in "
             "trajcontacts 1.0.1)",
    )
    cont.add_argument(
        "--condensed-out", dest="condensed_out", default=None, metavar="FILE",
        help="also write the upper triangle of the continuous matrix, one "
             "value per line in scipy squareform order, full precision",
    )

    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "-n", "--nproc", dest="nproc", type=int, default=1, metavar="N",
        help="worker processes (default: %(default)s). The distance kernel is "
             "already vectorised, so extra processes help mainly on very "
             "large systems.",
    )
    perf.add_argument(
        "--no-prefilter", dest="prefilter", action="store_false",
        help="disable the geometric pair prefilter. The prefilter is exact "
             "and normally a large speedup; this switch exists for debugging.",
    )
    perf.add_argument(
        "--memory", dest="memory_gb", type=float, default=2.0, metavar="GB",
        help="approximate scratch-memory budget used to size internal chunks "
             "(default: %(default)s)",
    )
    perf.add_argument(
        "-q", "--quiet", action="store_true", help="suppress progress output",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "-o", "--contact", dest="adjacency_out", default="contact.dat",
        metavar="FILE",
        help="unweighted adjacency matrix at the percent cutoff "
             "(default: %(default)s)",
    )
    out.add_argument(
        "-x", "--contactname", dest="pairs_out", default="contactResNames.dat",
        metavar="FILE",
        help="per-pair contact list with residue identification "
             "(default: %(default)s)",
    )
    out.add_argument(
        "-y", "--contactmatrixframes", dest="counts_out",
        default="contactMatrixResFrames.dat", metavar="FILE",
        help="matrix of frame counts (default: %(default)s)",
    )
    out.add_argument(
        "-z", "--contactmatrixfraction", dest="fraction_out",
        default="contactMatrixFraction.dat", metavar="FILE",
        help="weighted adjacency matrix of contact fractions "
             "(default: %(default)s)",
    )
    out.add_argument(
        "--numeric-out", dest="numeric_out", default="contact_num_only.dat",
        metavar="FILE",
        help="index-only pair list (default: %(default)s). This file was "
             "hardcoded in 0.1.x; pass 'none' to skip it.",
    )
    out.add_argument(
        "--npz", dest="npz_out", default=None, metavar="FILE",
        help="also write a compressed .npz archive of all results",
    )
    out.add_argument(
        "--min-fraction", dest="min_fraction", type=float, default=0.0,
        metavar="F",
        help="omit pairs seen in fewer than this fraction of frames from the "
             "pair lists (default: %(default)s, i.e. write every pair "
             "observed at least once). The matrices are unaffected.",
    )
    out.add_argument(
        "--legacy-format", action="store_true",
        help="write the pair lists using the 0.1.x column layout instead of "
             "the new labelled format",
    )
    out.add_argument(
        "--no-matrices", dest="write_matrices", action="store_false",
        help="skip the dense matrix files. Useful for large subsystems, where "
             "an N x N text matrix is the dominant cost.",
    )
    return parser


_TRAJECTORY_EXTENSIONS = {
    ".pdb", ".pdb.gz", ".xtc", ".trr", ".dcd", ".h5", ".hdf5", ".lh5",
    ".netcdf", ".nc", ".ncrst", ".crd", ".mdcrd", ".dtr", ".binpos",
    ".xyz", ".xyz.gz", ".gro", ".tng", ".rst7", ".restrt", ".xml",
    ".arc", ".lammpstrj",
}


def _looks_like_trajectory(path):
    lower = path.lower()
    return any(lower.endswith(ext) for ext in _TRAJECTORY_EXTENSIONS)


def _read_list_file(path):
    """Parse a text file of trajectory paths, one per line, '#' comments allowed.

    Relative entries are resolved against the list file's own directory, so
    the list can be moved together with the trajectories it names.
    """
    base = os.path.dirname(os.path.abspath(path))
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if not os.path.isabs(line) and not os.path.exists(line):
                line = os.path.join(base, line)
            entries.append(line)
    return entries


def resolve_trajectory_files(paths, parser):
    """Expand -f/--trajectory into what :func:`mdtraj.load` should receive.

    ``paths`` is one or more command-line arguments. Several arguments are
    taken as trajectory files to concatenate directly. A single argument
    whose extension is not a recognised trajectory format is instead read as
    a text file listing one trajectory path per line, so that many segments
    or replicates can be named without a very long command line.
    """
    if len(paths) > 1:
        return list(paths)

    path = paths[0]
    if _looks_like_trajectory(path):
        return path

    try:
        with open(path, "rb") as handle:
            chunk = handle.read(8192)
        chunk.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return path

    entries = _read_list_file(path)
    if not entries:
        return path

    missing = [e for e in entries if not os.path.exists(e)]
    if missing:
        parser.error(
            f"-f/--trajectory: {path!r} looks like a trajectory list file, "
            f"but lists missing file(s): {', '.join(missing)}"
        )
    return entries


def _validate(args, parser):
    if not os.path.exists(args.topology):
        parser.error(f"-p/--parmtop: file not found: {args.topology}")
    for path in args.trajectory:
        if not os.path.exists(path):
            parser.error(f"-f/--trajectory: file not found: {path}")
    if args.cutoff <= 0:
        parser.error("-c/--cutoff must be positive")
    if not 0.0 <= args.cutpercent <= 100.0:
        parser.error("-a/--cutpercent must be between 0 and 100")
    if args.min_separation < 1:
        parser.error("--min-separation must be >= 1")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.nproc < 1:
        parser.error("-n/--nproc must be >= 1")
    if not 0.0 <= args.min_fraction <= 1.0:
        parser.error("--min-fraction must be between 0 and 1")
    if args.mode in ("continuous", "both"):
        if args.sigma is not None:
            if not (args.sigma > 0 and np.isfinite(args.sigma)):
                parser.error("--sigma must be positive")
        else:
            if not (args.dmax > args.cutoff and np.isfinite(args.dmax)):
                parser.error("-d/--dmax must be larger than -c/--cutoff")
            if not 0.0 < args.kval < 1.0:
                parser.error("-k/--kval must be between 0 and 1 (exclusive)")
        if not 0 <= args.cont_precision <= 17:
            parser.error("--cont-precision must be between 0 and 17")


def _log(quiet, message):
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(args, parser)

    # Imported here so that --help and --version stay fast.
    import mdtraj as md
    from .core import (
        compute_contact_counts,
        compute_contacts_both,
        compute_continuous_contacts,
        heavy_atom_indices,
        make_pairs,
    )
    from . import io as tc_io

    started = time.time()
    trajectory_input = resolve_trajectory_files(args.trajectory, parser)
    if isinstance(trajectory_input, list) and len(trajectory_input) > 1:
        _log(
            args.quiet,
            f"Reading {len(trajectory_input)} trajectory files "
            f"(topology: {args.topology}) ...",
        )
        for path in trajectory_input:
            _log(args.quiet, f"  {path}")
    else:
        _log(args.quiet, f"Reading {trajectory_input} (topology: {args.topology}) ...")
    traj = md.load(trajectory_input, top=args.topology, stride=args.stride)

    end = args.end if args.end is not None else traj.n_frames
    if args.begin or end != traj.n_frames:
        begin = max(0, args.begin)
        traj = traj[begin:end]
    if traj.n_frames == 0:
        parser.error("no frames selected; check --begin/--end/--stride")

    selection = traj.topology.select(args.selection)
    if len(selection) == 0:
        parser.error(
            f"selection {args.selection!r} matched no atoms. "
            "Try -s all, or check the selection syntax at "
            "https://www.mdtraj.org/latest/atom_selection.html"
        )
    if len(selection) < traj.n_atoms:
        traj = traj.atom_slice(selection)

    topology = traj.topology
    n_res = topology.n_residues
    _log(
        args.quiet,
        f"{traj.n_frames} frames, {traj.n_atoms} atoms, {n_res} residues after "
        f"selection {args.selection!r}",
    )

    heavy = heavy_atom_indices(topology)
    empty = np.array([idx.size == 0 for idx in heavy])
    if empty.any():
        _log(
            args.quiet,
            f"note: {int(empty.sum())} residue(s) contain no heavy atoms and "
            "are excluded from the pair list",
        )

    pairs = make_pairs(n_res, args.min_separation, skippable=empty)
    if len(pairs) == 0:
        parser.error(
            "no residue pairs to analyse; check -s/--select and --min-separation"
        )
    _log(args.quiet, f"{len(pairs)} candidate residue pairs")

    if args.write_matrices and n_res > 20000:
        _log(
            args.quiet,
            f"warning: {n_res} residues will produce {n_res}x{n_res} text "
            "matrices. Consider -s/--select or --no-matrices.",
        )

    def progress(done, total):
        if not args.quiet:
            print(f"\r  frame chunk {done}/{total}", end="", file=sys.stderr, flush=True)
            if done == total:
                print("", file=sys.stderr, flush=True)

    common = dict(
        periodic=args.periodic,
        n_processes=args.nproc,
        prefilter=args.prefilter,
        memory_budget=int(args.memory_gb * 1e9),
        progress=progress,
    )
    continuous_kwargs = dict(
        sigma_nm=None if args.sigma is None else args.sigma / 10.0,
        d_max_nm=args.dmax / 10.0,
        k_at_d_max=args.kval,
        legacy_rounding=args.legacy_rounding,
    )
    result = continuous = None
    if args.mode == "binary":
        result = compute_contact_counts(
            traj, pairs, cutoff_nm=args.cutoff / 10.0, **common
        )
    elif args.mode == "continuous":
        continuous = compute_continuous_contacts(
            traj, pairs, cutoff_nm=args.cutoff / 10.0,
            **continuous_kwargs, **common,
        )
    else:
        result, continuous = compute_contacts_both(
            traj, pairs, cutoff_nm=args.cutoff / 10.0,
            **continuous_kwargs, **common,
        )

    fraction_cutoff = args.cutpercent / 100.0
    if result is not None:
        n_observed = int((result.counts > 0).sum())
        n_stable = int((result.fractions >= fraction_cutoff).sum())
        _log(
            args.quiet,
            f"{n_observed} pairs in contact at least once; "
            f"{n_stable} present in >= {args.cutpercent:g}% of frames",
        )
    if continuous is not None:
        n_weighted = int((continuous.sums > 0).sum())
        _log(
            args.quiet,
            f"continuous: sigma = {continuous.sigma_nm * 10.0:.6g} A"
            f"{', 1.0.1 rounding' if continuous.legacy_rounding else ''}; "
            f"{n_weighted} pairs with non-zero mean weight",
        )

    written = []
    if result is not None:
        written.extend(_write_binary(args, tc_io, result, fraction_cutoff))
    if continuous is not None:
        if (
            args.write_matrices
            and args.cont_matrix_out
            and args.cont_matrix_out.lower() != "none"
        ):
            tc_io.write_continuous_matrix(
                args.cont_matrix_out, continuous,
                fmt=f"%.{args.cont_precision}f",
            )
            written.append(args.cont_matrix_out)
        if args.condensed_out:
            tc_io.write_condensed(args.condensed_out, continuous)
            written.append(args.condensed_out)
    if args.npz_out:
        tc_io.write_npz(
            args.npz_out, result, fraction_cutoff, continuous=continuous
        )
        written.append(args.npz_out)

    _log(args.quiet, "wrote: " + ", ".join(written))
    _log(args.quiet, f"done in {time.time() - started:.1f} s")
    return 0


def _write_binary(args, tc_io, result, fraction_cutoff):
    """Write the binary-contact outputs; returns the paths written."""
    written = []
    if args.pairs_out and args.pairs_out.lower() != "none":
        writer = (
            tc_io.write_legacy_pair_table if args.legacy_format
            else tc_io.write_pair_table
        )
        writer(args.pairs_out, result, args.min_fraction)
        written.append(args.pairs_out)
    if args.numeric_out and args.numeric_out.lower() != "none":
        tc_io.write_legacy_numeric_table(args.numeric_out, result, args.min_fraction)
        written.append(args.numeric_out)
    if args.write_matrices:
        tc_io.write_matrices(
            result, fraction_cutoff,
            args.counts_out, args.fraction_out, args.adjacency_out,
        )
        written.extend(
            p for p in (args.counts_out, args.fraction_out, args.adjacency_out) if p
        )
    return written


if __name__ == "__main__":
    sys.exit(main())
