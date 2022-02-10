#!/usr/bin/env python

"""
Constants for unit conversion.

The internal reference units are for lengths and energy are Angstrom and
eV. All other units are context dependent.

"""

__author__ = "Alexander Urban"
__date__ = "2013-10-16"

# length
Bohr2Ang = 0.52917725
Ang2Bohr = 1.0/Bohr2Ang

# mass
me2kg = 9.1095e-31
kg2me = 1.0/me2kg
amu2kg = 1.6605e-27
kg2amu = 1.0/amu2kg

# Avogadro number (particles per mole)
N_A = mol = 6.02214129e23

# charge
e2C = 1.602176565e-19
C2e = 1.0/e2C

# energy
Ha2Ry = 2.0
Ha2eV = 27.2113961
Ry2Ha = 0.5
eV2J = 1.602176565e-19
kcal2J = 4.1868e3
Ry2eV = Ry2Ha*Ha2eV
eV2Ha = 1.0/Ha2eV
eV2Ry = 1.0/Ry2eV
J2eV = 1.0/eV2J
kcal2eV = kcal2J*J2eV
eV2kcal = 1.0/kcal2eV
eV2kcal_mol = eV2kcal*mol
kcal_mol2eV = 1.0/eV2kcal_mol
eV2kJ_mol = eV2J*mol*1.0e-3
kJ_mol2eV = 1.0/eV2kJ_mol

# time
Ha2fs = 0.02418884
fs2Ha = 1.0/Ha2fs

# frequency (icm = 1/cm)
eV2icm = 8066.04
icm2eV = 1.0/eV2icm
Ha2icm = 219488.0
icm2Ha = 1.0/Ha2icm
