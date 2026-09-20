"""trajcontacts2: residue-residue contacts from MD trajectories and PDB structures.

A pair of residues is defined to be in contact in a given frame when the
shortest distance between any two of their non-hydrogen atoms falls within a
cutoff (4-5 A). Contacts that persist for a majority of the simulation time
(75% by default) form the unweighted adjacency matrix used for downstream
network analysis.
"""

__version__ = "0.3.0"

from .core import (  # noqa: F401
    ContactResult,
    ResidueInfo,
    compute_contact_counts,
    describe_residues,
    heavy_atom_indices,
    make_pairs,
)

__all__ = [
    "__version__",
    "ContactResult",
    "ResidueInfo",
    "compute_contact_counts",
    "describe_residues",
    "heavy_atom_indices",
    "make_pairs",
]
