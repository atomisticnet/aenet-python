"""
Provides static data.

atomic_species[Z]['symbol'] Atomic symbol of species with atomic number Z
atomic_species[Z]['mass']   Atomic mass (g/mol) of species with atomic number Z

atomic_number[S]   Atomic number of species S

shannon_radii[S]   Ionic Shannon radii for species S
Can be queried using the function 'ionic_radius()'.

SCATTERING_FACTOR_COEFFICIENTS[S]  Cromer-Mann coefficients for the calculation
                                   of the atomic scattering factor of species S

"""

import json
from importlib import resources
import numpy as np

__author__ = "Alexander urban"
__date__ = "2013-03-25"

package = __package__

atomic_species = json.loads(
    resources.files(package).joinpath("data/species.json").read_text())

shannon_radii = json.loads(
    resources.files(package).joinpath(
        "data/ShannonRadii.json").read_text())

SCATTERING_FACTOR_COEFFICIENTS = json.loads(
    resources.files(package).joinpath(
        "data/AtomicScatteringFactors.json").read_text())

atomic_number = {}
for i, sp in enumerate(atomic_species):
    atomic_number[sp['symbol']] = i+1


def ionic_radius(species, oxidation_state, coordination=None, spin_state=None):
    """
    Return the ionic (Shannon) radius.

    Arguments:
      species (str)          Chemical symbol of the atomic species
      oxidation_state (int)  Atomic valence state
      coordination (int)     Coordination number
      spin_state             HS for high-spin or LS for low-spin

    Returns:
      radii, properties
      radii       list of ionic radii in Angstrom
      properties  dict with properties of each oxidation state

    """
    if species not in shannon_radii:
        raise ValueError("No ionic radii for {} in table.".format(species))
    states = [s for s in shannon_radii[species]
              if s['oxidation_state'] == oxidation_state]
    if coordination is not None:
        states = [s for s in states if s['coordination'] == coordination]
    if spin_state is not None:
        states = [s for s in states
                  if spin_state.lower() in s['spin_state'].lower()]
    radii = []
    properties = []
    for s in states:
        radii.append(s["ion"])
        properties.append(s)
    return radii, properties


def scattering_factor(species, wavelength, theta):
    """
    Return Cromer-Mann scattering factor for a given atomic species,
    radiation wavelength, and diffraction angle.

    Arguments:
      species (str)       Atomic symbol
      wavelength (float)  Wavelength in Angstrom
      theta (float)       Diffraction angle in degrees

    """

    if species not in SCATTERING_FACTOR_COEFFICIENTS:
        raise(ValueError("No Cromer-Mann coefficients available for species "
                         "{}.".format(species)))
    else:
        c = np.array(SCATTERING_FACTOR_COEFFICIENTS[species])
    Z = atomic_number[species]
    s = np.sin(theta)/wavelength
    s2 = s*s
    f = Z - 41.78214*s2*np.sum(c[:, 0]*np.exp(-c[:, 1]*s2))
    return f
