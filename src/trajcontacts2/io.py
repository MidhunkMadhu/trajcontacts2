"""Writers for the trajcontacts output files."""

from __future__ import annotations

import numpy as np

__all__ = [
    "write_pair_table",
    "write_legacy_pair_table",
    "write_legacy_numeric_table",
    "write_matrices",
    "write_npz",
]

_PAIR_HEADER = (
    "# trajcontacts residue contact list\n"
    "# cutoff = {cutoff:.3f} nm ({cutoff_ang:.2f} A), frames = {n_frames}\n"
    "# index_i and index_j are 0-based indices into the selected subsystem;\n"
    "# chain/resSeq/resName reproduce the labels of the input topology.\n"
    "#{i:>7s} {j:>8s} {ci:>6s} {si:>8s} {ni:>6s} {cj:>6s} {sj:>8s} {nj:>6s} "
    "{nf:>9s} {fr:>9s}\n"
)


def _iter_rows(result, min_fraction):
    fractions = result.fractions
    residues = result.residues
    for k in range(len(result.pairs)):
        fraction = fractions[k]
        if fraction < min_fraction:
            continue
        i, j = int(result.pairs[k, 0]), int(result.pairs[k, 1])
        yield i, j, residues[i], residues[j], int(result.counts[k]), float(fraction)


def write_pair_table(path, result, min_fraction=0.0):
    """Write the per-pair contact list with full residue identification."""
    with open(path, "w") as handle:
        handle.write(
            _PAIR_HEADER.format(
                cutoff=result.cutoff_nm,
                cutoff_ang=result.cutoff_nm * 10.0,
                n_frames=result.n_frames,
                i="index_i",
                j="index_j",
                ci="chain_i",
                si="resSeq_i",
                ni="res_i",
                cj="chain_j",
                sj="resSeq_j",
                nj="res_j",
                nf="n_frames",
                fr="fraction",
            )
        )
        for i, j, ri, rj, count, fraction in _iter_rows(result, min_fraction):
            handle.write(
                f"{i:8d} {j:8d} {ri.chain:>6s} {ri.res_seq:8d} {ri.name:>6s} "
                f"{rj.chain:>6s} {rj.res_seq:8d} {rj.name:>6s} "
                f"{count:9d} {fraction:9.4f}\n"
            )


def write_legacy_pair_table(path, result, min_fraction=0.0):
    """Reproduce the 0.1.x ``contactResNames.dat`` layout.

    Columns: index_i, label_i, index_j, label_j, n_frames, fraction.  The
    header is corrected (0.1.x printed ``res1_index`` twice and omitted the
    fraction column) but the data columns are unchanged.
    """
    with open(path, "w") as handle:
        handle.write(
            "#res1_index   res2_index    res1    res2     "
            "n_frames_contact    fraction\n"
        )
        for i, j, ri, rj, count, fraction in _iter_rows(result, min_fraction):
            handle.write(
                f"{i} {ri.name}{ri.res_seq} {j} {rj.name}{rj.res_seq} "
                f"{count} {round(fraction, 2)}\n"
            )


def write_legacy_numeric_table(path, result, min_fraction=0.0):
    """Reproduce the 0.1.x ``contact_num_only.dat`` layout."""
    with open(path, "w") as handle:
        for i, j, _ri, _rj, count, fraction in _iter_rows(result, min_fraction):
            handle.write(f"{i} {j} {count} {round(fraction, 2)}\n")


def write_matrices(result, fraction_cutoff, counts_path, fraction_path, adjacency_path):
    """Write the three dense matrices, matching the 0.1.x formats."""
    counts = result.count_matrix()
    if counts_path:
        np.savetxt(counts_path, counts, fmt="%d")
    if fraction_path:
        np.savetxt(fraction_path, result.fraction_matrix(), fmt="%.2f")
    if adjacency_path:
        np.savetxt(
            adjacency_path, result.adjacency_matrix(fraction_cutoff), fmt="%d"
        )


def write_npz(path, result, fraction_cutoff):
    """Compressed binary archive: pairs, counts, fractions and residue labels."""
    residues = result.residues
    np.savez_compressed(
        path,
        pairs=result.pairs,
        counts=result.counts,
        fractions=result.fractions,
        adjacency=result.adjacency_matrix(fraction_cutoff),
        n_frames=np.asarray(result.n_frames),
        cutoff_nm=np.asarray(result.cutoff_nm),
        fraction_cutoff=np.asarray(fraction_cutoff),
        residue_index=np.asarray([r.index for r in residues]),
        residue_chain=np.asarray([r.chain for r in residues]),
        residue_seq=np.asarray([r.res_seq for r in residues]),
        residue_name=np.asarray([r.name for r in residues]),
    )
