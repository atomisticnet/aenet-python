"""
Auxiliary routines used in multiple places in the package.

Note: Geometry-specific utilities have been moved to aenet.geometry.utils
"""

import numpy as np
import scipy.stats
from contextlib import contextmanager
import os

__author__ = "Alexander urban"
__date__ = "2013-03-25"


def decompose_composition(comp, basis):
    """
    Decompose a composition into elements or other compositions.

    Arguments:
      comp (dict)   a composition dictionary, e.g., { 'H' : 2, 'O' : 1 }
      basis (list)  a list containing either chemical symbols of elements
                    or other composition dictionaries,
                    e.g. ['H', {'H': 1, 'O' : 1 }]
                    Diatomic elements (e.g., 'H2') are currently not
                    supported and have to be entered as dict {'H':2}.

    Returns tuple (coeff, coeff/|coeff|)
      where coeff is an ndarray with the expansion coefficients
    """
    # make sure all basis functions are dictionaries: 'A' --> {'A' : 1}
    species = []
    for i in range(len(basis)):
        if not isinstance(basis[i], dict):
            basis[i] = {basis[i]: 1}
        species = species + list(basis[i].keys())
    species = list(set(species))
    for s in comp:
        if s not in species:
            raise ValueError("Composition can not be represented by basis")

    # express each basis function as normalized vector in species space
    def species_vector(c):
        v = []
        for s in species:
            if s in c:
                v.append(c[s])
            else:
                v.append(0.0)
        v = np.array(v, dtype=np.double)
        return v

    bvec = []
    for b in basis:
        v = species_vector(b)
        # bvec.append(v/np.linalg.norm(v))
        bvec.append(v)
    bvec = np.array(bvec)
    det = np.linalg.det(bvec)
    if det == 0.0:
        raise ValueError("Basis compositions are not independent")
    # expand input composition in basis
    compvec = species_vector(comp)
    coeff = np.linalg.solve(bvec.T, compvec)
    return coeff, coeff/np.linalg.norm(coeff)


def compute_moments(array, moment=1, axis=0):
    """
    Helper function to compute moments of a list up to a degree.
    Default is the mean.
    To reduce down a 2D array to a single moment, run the function twice.

    Arguments:
      array    single vector (list) or matrix (list of lists)
      moment   up to which moment to compute (positive int)
      axis     what axis to use for moment computation; consult numpy for info

    Returns:
      moments_list    list of moments
     """

    if not isinstance(moment, int) or moment < 1:
        raise ValueError('Moment must be a positive integer.')
    moments_list = np.array([np.mean(array, axis=0)])
    if moment > 1:
        moments_list = np.append(
            moments_list, scipy.stats.moment(
                array, moment=range(2, moment + 1), axis=0), axis=0)
    return moments_list


def trynumeric(string, use_pandas=False):
    """
    Try to convert a string to a numeric value.  If unsuccessful, return
    original string.

    Arguments:
      string (str): The input string that is to be converted
      use_pandas (bool): If True, use Pandas `eval` function to attempt
        conversion if simple conversion to int/float failed.

    """
    try:
        out = int(string)
    except ValueError:
        try:
            out = float(string)
        except ValueError:
            if use_pandas:
                try:
                    import pandas as pd
                    out = pd.eval(string)
                except ValueError:
                    out = string
            else:
                out = string
    return out


def csv2dict(csvlist):
    """
    Convert list of comma separated values to a dictionary.

    The hierarchy of the dictionary is expressed by equal signs ("=") and
    colons (":").  For example

       ["Co=Co:0.5,Ni:0.5", "Li=Na"]

    will be converted to

       {"Co": {"Co": 0.5, "Ni": 0.5}, "Li": "Na"}

    """
    def trynumeric(v):
        try:
            out = int(v)
        except ValueError:
            try:
                import pandas as pd
                out = pd.eval(v)
            except ValueError:
                out = v
        return out
    outdict = {}
    for item in csvlist:
        key, value = item.split("=")
        if ":" in value:
            value = {s.strip(): trynumeric(v.strip()) for s, v in
                     [el.split(":") for el in value.split(",")]}
        else:
            value = trynumeric(value.strip())
        outdict[key.strip()] = value
    return outdict


def csv2list(csvlist):
    """
    Convert list of comma separated values to am actual list.

    Ranges are defined by ":".  For example:

       [1, "2", "3:7", "8,9:11"]

    will be converted to

       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    """
    def trynumeric(v):
        try:
            out = int(v)
        except ValueError:
            try:
                import pandas as pd
                out = pd.eval(v)
            except ValueError:
                out = v
        return out

    def expand_range(r):
        if ":" not in r:
            return [int(r)]
        i0, i1 = [int(s) for s in r.split(":")]
        return list(range(i0, i1)) + [i1]

    ll = []
    for item in csvlist:
        try:
            ll += [int(item)]
        except ValueError:
            for item2 in item.split(","):
                ll += expand_range(item2)
    return ll


@contextmanager
def cd(newdir):
    """
    Change to a directory using context manager.

    Example:

      with cd(./tmp/):
        ...

    Source: https://stackoverflow.com/a/24176022/1013199

    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield {'origin': prevdir, 'current': os.getcwd()}
    finally:
        os.chdir(prevdir)


# Re-export geometry functions with deprecation warnings via __getattr__
_DEPRECATED_GEOMETRY_FUNCTIONS = [
    'standard_cell',
    'rotate',
    'rotation_matrix',
    'wrap_pbc',
    'cellmatrix_from_params',
    'transform_cell',
    'add_redundant_atoms',
    'del_redundant_atoms',
    'strain_orthorhombic',
    'strain_monoclinic',
    'symmetry_equivalent_atoms',
    'center_of_mass',
]


def __getattr__(name):
    """
    Provide deprecated access to geometry utility functions.
    """
    if name in _DEPRECATED_GEOMETRY_FUNCTIONS:
        import warnings
        from .geometry import utils as geometry_utils
        warnings.warn(
            f"Importing '{name}' from aenet.util is deprecated and "
            f"will be removed in a future version. "
            f"Use 'from aenet.geometry.utils import {name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return getattr(geometry_utils, name)
    raise AttributeError(f"module 'aenet.util' has no attribute '{name}'")
