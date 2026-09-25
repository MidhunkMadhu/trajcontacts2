"""trajcontacts2: residue-residue contacts from MD trajectories and PDB structures.

A pair of residues is defined to be in contact in a given frame when the
shortest distance between any two of their non-hydrogen atoms falls within a
cutoff (4-5 A). Contacts that persist for a majority of the simulation time
(75% by default) form the unweighted adjacency matrix used for downstream
network analysis.

A continuous ("semi-Gaussian") contact weight, 1 within the cutoff and
decaying smoothly beyond it, is available as an alternative or in addition
(trajcontacts 1.0.1 ``-m cont``; Westerlund et al. 2020).
"""

__version__ = "0.4.0"

from .core import (  # noqa: F401
    ContactResult,
    ContinuousContactResult,
    ResidueInfo,
    compute_contact_counts,
    compute_contacts_both,
    compute_continuous_contacts,
    continuous_sigma,
    describe_residues,
    heavy_atom_indices,
    kernel_support,
    make_pairs,
    semi_gaussian_kernel,
)

__all__ = [
    "__version__",
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
