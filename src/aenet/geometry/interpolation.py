#!/usr/bin/env python

"""
Structure interpolation utilities.

"""

import numpy as np

from ..exceptions import ArgumentError

__author__ = "Alexander Urban, Nongnuch Artrith"
__date__ = "2013-06-05"


def interpolate(s1, s2, n=1, s3=None):
    """
    Interpolate between the two structures s1 and s2.  Return n
    interpolated structures.  If only s1 and s2 are specified, the
    structures will be linearly interpolated.  If also s3 is given,
    all interpolated structures will be on a parabola through s3.

    Arguments:
      s1, s2    end points for interpolated path (AtomicStructure)
      s3        optional intermediate point (AtomicStructure)
      n         number of interpolated structures to be returned

    Returns:
      Instance of AtomicStructure with n+2 frames, where the first and
      the last frame correspond to s1 and s2.
    """
    # Import here to avoid circular dependency
    from .structure import AtomicStructure

    if ((s2.natoms != s1.natoms) or (s2.ntypes != s1.ntypes) or
            (s1.pbc != s2.pbc)):
        raise ArgumentError(
            "Incompatible start and end structures for interpolation.")
    if (s3) and ((s3.natoms != s1.natoms) or (s3.ntypes != s1.ntypes) or
                 (s3.pbc != s1.pbc)):
        raise ArgumentError(
            "Incompatible intermediate structure for interpolation.")

    if s1.pbc:
        s = AtomicStructure(s1.coords[-1], s1.types, avec=s1.avec[-1],
                            fixed=s1.fixed)
    else:
        s = AtomicStructure(s1.coords[-1], s1.types, fixed=s1.fixed)
    if not s3:
        # linear interpolation between s1 and s2
        for x in np.linspace(0.0, 1.0, n+2):
            if (x > 0.0):
                coords = (1.0-x)*s1.coords[-1] + x*s2.coords[-1]
                if s.pbc:
                    avec = (1.0-x)*s1.avec[-1] + x*s2.avec[-1]
                    s.add_frame(coords, avec=avec)
                else:
                    s.add_frame(coords)
    else:
        c0 = s1.coords[-1]
        c1 = -3.0*s1.coords[-1] - s2.coords[-1] + 4.0*s3.coords[-1]
        c2 = 2.0*s1.coords[-1] + 2.0*s2.coords[-1] - 4.0*s3.coords[-1]
        if s.pbc:
            a0 = s1.avec[-1]
            a1 = -3.0*s1.avec[-1] - s2.avec[-1] + 4.0*s3.avec[-1]
            a2 = 2.0*s1.avec[-1] + 2.0*s2.avec[-1] - 4.0*s3.avec[-1]
        # second order interpolation, assuming s3 is exactly half-way
        # from s1 to s2
        for x in np.linspace(0.0, 1.0, n+2):
            if (x > 0.0):
                coords = c0 + c1*x + c2*x*x
                if s.pbc:
                    avec = a0 + a1*x + a2*x*x
                    s.add_frame(coords, avec=avec)
                else:
                    s.add_frame(coords)

    return s
