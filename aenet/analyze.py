"""
Methods and classes related to analysis of atomic structures and
trajectories.

"""

import sys
import numpy as np
from scipy import stats
import multiprocessing as mp
import functools

from .nblist import NeighborList
from .exceptions import ArgumentError
from .staticdata import atomic_number

__author__ = "Alexander Urban and Nongnuch Artrith"
__date__ = "2013-09-28"


def coordination_number(struc, atoms=None, species=None, rcut=3.5,
                        cores=1, smooth=False, initial_frame=-1,
                        final_frame=None):
    """
    Calculate coordination numbers in a structure or over a trajectory.

    Arguments:
      struc          an AtomicStructure instance
      atoms (list)   list of integers with atomic indices starting with 1
                     if None, all atoms are considered
      species (list) list of coordinating atomic species; if None, all
                     species are considered
      rcut (float)   cutoff radius in Angstrom
      cores (int)    number of processes to use for computation
      smooth         smooth/continuous decay of the coordination number;
                     if True, rcut is taken to be a scaling parameter, d_AB,
                     such that the coordination number is defined as:

          c_AB = 1/N_A sum_i^N_A (sum_j^N_B (1 - s^6)/(1 - s^12))

                     with s = r_ij/d_AB

          where r_ij is the interatomic distance and d_AB is a scaling
          parameter for the species pair A-B.

          This definition of the coordination number was introduced in:
          M. Iannuzzi, A. Laio, and M. Parrinello, PRL 90 (2003) 238302.

      initial_frame (int):  initial frame to be analyzed
      final_frame (int):  final frame to be analyzed; defaults to the final
        trajectory frame

    """

    if atoms is None:
        atoms = range(1, struc.natoms+1)

    if species is None:
        species = list(set(struc.types))

    if initial_frame < 0:
        initial_frame = struc.nframes + initial_frame

    if final_frame is None:
        final_frame = struc.nframes - 1
    elif final_frame < 0:
        final_frame = struc.nframes + final_frame

    cn_for_frame = functools.partial(_coordination_number_for_frame,
                                     struc, rcut, atoms, species,
                                     smooth)

    frames = range(initial_frame, final_frame+1)
    if cores == 1:
        c_ij = np.array(list(map(cn_for_frame, frames)))
    else:
        pool = mp.Pool(processes=cores)
        c_ij = np.array(pool.map(cn_for_frame, frames))

    return c_ij


def _coordination_number_for_frame(struc, rcut, atoms, species, smooth, frame):
    """
    Evaluate coordination numbers for a single MD frame.

    """
    if smooth:
        interaction_range = 2*rcut

        def truncate(r_ij):
            s6 = (r_ij/rcut)**6
            return (1.0 - s6)/(1.0 - s6*s6)
    else:
        interaction_range = rcut

        def truncate(r_ij):
            return 1.0 if r_ij <= rcut else 0.0

    avec = None if not struc.pbc else struc.avec[frame]
    nbl = NeighborList(struc.coords[frame], lattice_vectors=avec,
                       cartesian=True, types=struc.types,
                       interaction_range=interaction_range)
    k = 0
    c_ij = np.zeros(len(atoms))
    for iatom in atoms:
        i = iatom - 1
        (nb, dist, T) = nbl.get_neighbors_and_distances(i)
        c_ij[k] = 0.0
        for j in range(len(nb)):
            if not struc.types[nb[j]] in species:
                continue
            r_ij = dist[j]
            c_ij[k] += truncate(r_ij)
        k += 1
    return c_ij


def bond_length(struc, atoms=None, species=None, rcut=3.5, cores=1,
                initial_frame=-1, final_frame=None):
    """
    Determine minimal bond length in a structure or over a trajectory.

    Arguments:
      struc          an AtomicStructure instance
      atoms (list)   list of integers with atomic indices starting with 1
                     if None, all atoms are considered
      species (list) list of coordinating atomic species; if None, all
                     species are considered
      rcut (float)   cutoff radius in Angstrom
      cores (int)    number of processes to use for computation
      initial_frame (int):  initial frame to be analyzed
      final_frame (int):  final frame to be analyzed; defaults to the final
        trajectory frame

    """

    if atoms is None:
        atoms = range(1, struc.natoms+1)

    if species is None:
        species = list(set(struc.types))

    if initial_frame < 0:
        initial_frame = struc.nframes + initial_frame

    if final_frame is None:
        final_frame = struc.nframes - 1
    elif final_frame < 0:
        final_frame = struc.nframes + final_frame

    bl_for_frame = functools.partial(_bond_length_for_frame,
                                     struc, rcut, atoms, species)

    frames = range(initial_frame, final_frame+1)
    if cores == 1:
        d_ij = np.array(list(map(bl_for_frame, frames)))
    else:
        pool = mp.Pool(processes=cores)
        d_ij = np.array(pool.map(bl_for_frame, frames))

    return d_ij


def _bond_length_for_frame(struc, rcut, atoms, species, frame):
    """
    Evaluate bond length for a single MD frame.

    """
    interaction_range = rcut
    avec = None if not struc.pbc else struc.avec[frame]
    nbl = NeighborList(struc.coords[frame], lattice_vectors=avec,
                       cartesian=True, types=struc.types,
                       interaction_range=interaction_range)
    k = 0
    d_ij = np.zeros(len(atoms))
    for iatom in atoms:
        i = iatom - 1
        (nb, dist, T) = nbl.get_neighbors_and_distances(i)
        d_ij[k] = np.min([d for j, d in enumerate(dist)
                          if struc.types[nb[j]] in species])
        k += 1
    return d_ij


def mean_squared_displacement(struc, dt, first=1, last=None, atoms=None,
                              outfile=None, avrange=None, t0=0.0,
                              total=False, direct=False, max_avg=False,
                              cores=1):
    """
    Calculate the mean squared displacement of all frames from the
    structure in the first frame of an MD trajectory.

    Arguments:
      struc    instance of the AtomicStructure class
      dt       MD timestep in femtoseconds (output in picoseconds)
      first    first frame (reference structure; default: 1)
      last     last frame to consider (default: None)
      atoms    list of atoms to average over; if None, all atoms will
               be used; 1 <= atoms[i] <= struc.natoms
      outfile  name of an output filel if None, write to stdout
      avrange  Number of MD steps to average over to determine the MSD;
               if avrange is not specified, it will be set to struc.nframes/2
      t0       time corresponding to the first frame
      total    compute the total squared displacement, not the MSD
               i.e., return <(sum_i r_i)^2> not sum <r_i^2>
      direct   just compute direct displacement relative to reference frame;
               no averaging will be done, so trajectory time is preserved
      max_avg  if True, use maximal possible number of averages to compute
               the ensemble average at every time difference;  Per default,
               a constant number of avrange averages are used, and only
               those time differences that have N/2 samples are considered.
      cores    number of parallel processes to be used for computation
    """

    if (abs(first) > struc.nframes) or (
            last is not None and abs(last) > struc.nframes):
        raise ArgumentError("Specified frame range is invalid.")

    if atoms is None:
        atoms = range(1, struc.natoms+1)

    for i in atoms:
        if (i > struc.natoms) or (i < 1):
            raise ArgumentError("Invalid atom indices.")

    if last is None:
        last = struc.nframes

    if max_avg:
        avrange = last - first
    elif avrange is None:
        avrange = int((last - first + 1)/2)
    elif avrange > (last - first + 1)/2:
        sys.stderr("Warning: Averaging interval greater than nframes/2. "
                   "The ensemble average becomes unreliable at large t.\n")

    # first frame is actually frame 0:
    first -= 1
    last -= 1

    # convert fs to ps:
    t0 = t0/1000.0
    dt = dt/1000.0

    if outfile is None:
        f = sys.stdout
    else:
        f = open(outfile, 'w')

    f.write("#    Mean Squared Displacement (MSD) for "
            "{} MD frames and {} atoms\n".format(last-first+1, len(atoms)))
    f.write("#    t (ps)      ")
    if total:
        f.write("STD (Ang^2)     STD_x (Ang^2)   STD_y (Ang^2)   "
                "STD_z (Ang^2) \n")
    else:
        f.write("MSD (Ang^2)     MSD_x (Ang^2)   MSD_y (Ang^2)   "
                "MSD_z (Ang^2) \n")

    nframes = last - first + 1

    # cm^2/s = 10^-4 m^2/s
    # A^2/ps = 10^-8 m^2/s = 10^-4 cm^2/s
    # d3 = 1.0/(6.0*dt)*1.0e-4
    # d1 = 1.0/(2.0*dt)*1.0e-4
    d3 = 1.0/6.0*1.0e-4
    d1 = 1.0/2.0*1.0e-4

    struc.align_all_frames()
    if direct:
        t = np.arange(0.0, dt*nframes, dt)
        norm = 1.0/len(atoms)
        MSD = np.zeros(nframes)
        MSD_x = np.zeros(nframes)
        MSD_y = np.zeros(nframes)
        MSD_z = np.zeros(nframes)
        if total:
            msd_for_frame = functools.partial(_msd_for_frame, struc,
                                              atoms, nframes, first, 0)
        else:
            msd_for_frame = functools.partial(_std_for_frame, struc,
                                              atoms, nframes, first, 0)
        (MSD, MSD_x, MSD_y, MSD_z) = msd_for_frame(0)
        MSD *= norm
        MSD_x *= norm
        MSD_y *= norm
        MSD_z *= norm
        # write out MSD for all time differences
        for i in range(nframes):
            f.write(("{:15.8f}" + 4*" {:15.8e}" + "\n").format(
                t[i], MSD[i], MSD_x[i], MSD_y[i], MSD_z[i]))
        # compute diffusivity by linear regression
        slope, intercept, r, p, stderr = stats.linregress(t, MSD)
        D = d3*slope
        slope, intercept, r, p, stderr = stats.linregress(t, MSD_x)
        D_x = d1*slope
        slope, intercept, r, p, stderr = stats.linregress(t, MSD_y)
        D_y = d1*slope
        slope, intercept, r, p, stderr = stats.linregress(t, MSD_z)
        D_z = d1*slope
        # write out diffusivity
        f.write(("# D, Dx, Dy, Dz (cm^2/s) = " + 4*" {:15.8e}" +
                 "\n").format(D, D_x, D_y, D_z))
        f.write("\n\n")
    else:
        t = np.arange(0.0, dt*avrange, dt)
        MSD = np.zeros(avrange)
        MSD_x = np.zeros(avrange)
        MSD_y = np.zeros(avrange)
        MSD_z = np.zeros(avrange)
        if max_avg:
            norm = 1.0/len(atoms)
            Nav = 1
        else:
            norm = 1.0/len(atoms)/avrange
            Nav = int(nframes/avrange) - 1
        for j in range(Nav):
            if total:
                msd_for_frame = functools.partial(_msd_for_frame, struc,
                                                  atoms, avrange, first, j)
            else:
                msd_for_frame = functools.partial(_std_for_frame, struc,
                                                  atoms, avrange, first, j)
            if cores == 1:
                results = [msd_for_frame(el) for el in range(avrange)]
            else:
                pool = mp.Pool(processes=cores)
                results = pool.map(msd_for_frame, range(avrange))

            MSD[:] = 0.0
            MSD_x[:] = 0.0
            MSD_y[:] = 0.0
            MSD_z[:] = 0.0
            for (xyz, x, y, z) in results:
                MSD[:] += xyz
                MSD_x[:] += x
                MSD_y[:] += y
                MSD_z[:] += z

            MSD *= norm
            MSD_x *= norm
            MSD_y *= norm
            MSD_z *= norm

            if max_avg:
                samples = avrange - 1
                for i in range(avrange-1):
                    MSD[i] /= samples
                    MSD_x[i] /= samples
                    MSD_y[i] /= samples
                    MSD_z[i] /= samples
                    samples -= 1

            # write out MSD for all time differences
            for i in range(avrange):
                f.write(("{:15.8f}" + 4*" {:15.8e}" + "\n").format(
                    t[i], MSD[i], MSD_x[i], MSD_y[i], MSD_z[i]))
            # compute diffusivity by linear regression
            slope, intercept, r, p, stderr = stats.linregress(t, MSD)
            D = d3*slope
            slope, intercept, r, p, stderr = stats.linregress(t, MSD_x)
            D_x = d1*slope
            slope, intercept, r, p, stderr = stats.linregress(t, MSD_y)
            D_y = d1*slope
            slope, intercept, r, p, stderr = stats.linregress(t, MSD_z)
            D_z = d1*slope
            # write out diffusivity
            f.write(("# D, Dx, Dy, Dz (cm^2/s) = " + 4*" {:15.8e}" +
                     "\n").format(D, D_x, D_y, D_z))
            f.write("\n\n")

    if outfile is not None:
        f.close()


def _msd_for_frame(struc, atoms, avrange, first, j, i):
    """
    Compute Mean Squared Displacement for a single MD frame.  Actually,
    this routine returns the sum of squared displacements, which can
    then be normalized to the MSD.

    """
    SSD_x = np.zeros(avrange)
    SSD_y = np.zeros(avrange)
    SSD_z = np.zeros(avrange)
    SSD = np.zeros(avrange)
    if1 = first + i + j*avrange
    samples = min(avrange, struc.nframes - if1)
    for k in range(1, samples):
        if2 = if1 + k
        dx2 = dy2 = dz2 = 0.0
        for at in atoms:
            dx = struc.coords[if2][at-1][0] - struc.coords[if1][at-1][0]
            dy = struc.coords[if2][at-1][1] - struc.coords[if1][at-1][1]
            dz = struc.coords[if2][at-1][2] - struc.coords[if1][at-1][2]
            dx2 += dx*dx
            dy2 += dy*dy
            dz2 += dz*dz
        # update sum of squared displacements
        SSD_x[k] = dx2
        SSD_y[k] = dy2
        SSD_z[k] = dz2
        SSD[k] = (dx2 + dy2 + dz2)
    return (SSD, SSD_x, SSD_y, SSD_z)


def _std_for_frame(struc, atoms, avrange, first, j, i):
    """
    Compute squared total displacement for a single MD frame.  Actually,
    this routine returns the square of the sum of the atomic
    displacements, which can then be normalized to the squared total
    displacement.

    """
    SSD_x = np.zeros(avrange)
    SSD_y = np.zeros(avrange)
    SSD_z = np.zeros(avrange)
    SSD = np.zeros(avrange)
    if1 = first + i + j*avrange
    samples = min(avrange, struc.nframes - if1)
    for k in range(1, samples):
        if2 = if1 + k
        dx2 = dy2 = dz2 = 0.0
        for at in atoms:
            dx = struc.coords[if2][at-1][0] - struc.coords[if1][at-1][0]
            dy = struc.coords[if2][at-1][1] - struc.coords[if1][at-1][1]
            dz = struc.coords[if2][at-1][2] - struc.coords[if1][at-1][2]
            dx2 += dx
            dy2 += dy
            dz2 += dz
        dx2 = dx2*dx2
        dy2 = dy2*dy2
        dz2 = dz2*dz2
        # update sum of squared displacements
        SSD_x[k] = dx2
        SSD_y[k] = dy2
        SSD_z[k] = dz2
        SSD[k] = (dx2 + dy2 + dz2)
    return (SSD, SSD_x, SSD_y, SSD_z)


def mean_squared_displacement_OLD(struc, dt, first=1, last=None, atoms=None,
                                  outfile=None, avrange=100, t0=0.0):
    """
    Calculate the mean squared displacement of all frames from the
    structure in the first frame of an MD trajectory.

    Arguments:
      struc    instance of the AtomicStructure class
      dt       MD timestep in femtoseconds (output in picoseconds)
      first    first frame (reference structure; default: 1)
      last     last frame to consider (default: None)
      atoms    list of atoms to average over; if None, all atoms will
               be used; 1 <= atoms[i] <= struc.natoms
      outfile  name of an output filel if None, write to stdout
      avrange  Number of MD steps to average over to determine the diffusivity;
               the reference frame for the MSD will also be updated according
               to this interval
      t0       time corresponding to the first frame
    """

    if (abs(first) > struc.nframes) or (
            last is not None and abs(last) > struc.nframes):
        raise ArgumentError("Specified frame range is invalid.")

    if atoms is None:
        atoms = range(1, struc.natoms+1)

    for i in atoms:
        if (i > struc.natoms) or (i < 1):
            raise ArgumentError("Invalid atom indices.")

    if last is None:
        last = struc.nframes

    # first frame is actually frame 0:
    first -= 1
    last -= 1

    # convert fs to ps:
    t0 = t0/1000.0
    dt = dt/1000.0

    if outfile is None:
        f = sys.stdout
    else:
        f = open(outfile, 'w')

    f.write("#    Mean Squared Displacement (MSD) for "
            "{} MD frames and {} atoms\n".format(last-first+1, len(atoms)))
    f.write("#    t (ps)      ")
    f.write("MSD (Ang^2)     MSD_x (Ang^2)   MSD_y (Ang^2)   MSD_z (Ang^2)   ")
    f.write("D (cm^2/s)      D_x (cm^2/s)    D_y (cm^2/s)    D_z (cm^2/s) \n")

    f.write(("{:15.8f}" + 4*" {:15.8e}" + "\n").format(
        t0, 0, 0, 0, 0))

    # cm^2/s = 10^-4 m^2/s
    # A^2/ps = 10^-8 m^2/s = 10^-4 cm^2/s
    # d3 = 1.0/(6.0*dt)*1.0e-4
    # d1 = 1.0/(2.0*dt)*1.0e-4
    d3 = 1.0/6.0*1.0e-4
    d1 = 1.0/2.0*1.0e-4
    MSD_x_list = np.zeros(avrange)
    MSD_y_list = np.zeros(avrange)
    MSD_z_list = np.zeros(avrange)
    MSD_list = np.zeros(avrange)
    t_list = np.zeros(avrange)
    t_list[0] = t0

    norm = 1.0/len(atoms)
    MSD_x = 0.0
    MSD_y = 0.0
    MSD_z = 0.0
    MSD = 0.0
    t = t0/1000.0 + dt
    ref = first
    MSD0 = 0.0
    MSD0_x = 0.0
    MSD0_y = 0.0
    MSD0_z = 0.0
    for i in range(struc.nframes)[first+1:last+1]:
        if i - ref + 1 > avrange:
            t0 = t - dt
            MSD0 += MSD
            MSD0_x += MSD_x
            MSD0_y += MSD_y
            MSD0_z += MSD_z
            t_list[0] = t0
            MSD_list[0] = 0.0
            MSD_x_list[0] = 0.0
            MSD_y_list[0] = 0.0
            MSD_z_list[0] = 0.0
            ref = i - 1
        struc.align_frames(i, i-1)
        dx2 = dy2 = dz2 = 0.0
        for j in atoms:
            dx = struc.coords[i][j-1][0] - struc.coords[ref][j-1][0]
            dy = struc.coords[i][j-1][1] - struc.coords[ref][j-1][1]
            dz = struc.coords[i][j-1][2] - struc.coords[ref][j-1][2]
            dx2 += norm*dx*dx
            dy2 += norm*dy*dy
            dz2 += norm*dz*dz
        # update MSD
        MSD_x = dx2
        MSD_y = dy2
        MSD_z = dz2
        MSD = (dx2+dy2+dz2)
        # remember all MSD values for the averaging window
        t_list[i-ref] = t
        MSD_list[i-ref] = MSD
        MSD_x_list[i-ref] = dx2
        MSD_y_list[i-ref] = dy2
        MSD_z_list[i-ref] = dz2
        # determine diffusivity from the slope over averaging window
        if i - ref >= 2:
            slope, intercept, r, p, stderr = stats.linregress(
                t_list[:(i-ref+1)], MSD_list[:(i-ref+1)])
            D = d3*slope
            slope, intercept, r, p, stderr = stats.linregress(
                t_list[:(i-ref+1)], MSD_x_list[:(i-ref+1)])
            D_x = d1*slope
            slope, intercept, r, p, stderr = stats.linregress(
                t_list[:(i-ref+1)], MSD_y_list[:(i-ref+1)])
            D_y = d1*slope
            slope, intercept, r, p, stderr = stats.linregress(
                t_list[:(i-ref+1)], MSD_z_list[:(i-ref+1)])
            D_z = d1*slope
            f.write(("{:15.8f}" + 8*" {:15.8e}" + "\n").format(
                t, MSD+MSD0, MSD_x+MSD0_x, MSD_y+MSD0_y, MSD_z+MSD0_z,
                D, D_x, D_y, D_z))
        else:
            f.write(("{:15.8f}" + 4*" {:15.8e}" + "\n").format(
                t, MSD+MSD0, MSD_x+MSD0_x, MSD_y+MSD0_y, MSD_z+MSD0_z))

        t = t + dt

    if outfile is not None:
        f.close()


def radial_pair_distribution(struc, length, first=1, last=None,
                             nbins=100, species=None, atoms=None,
                             cores=1, normalize=True,
                             scattering_factors=False):
    """
    Calculate the radial pair distribution function (RDF) of a single
    structure or an MD trajectory.

    Arguments:
      struc      an AtomicStructure instance
      length     the range of the RDF
      first      first MD frame to be used
      last       last MD frame to be used
      nbins      number of bins to compute the histogram
      species    if not None, a tuple of two atomic species (A, B);
                 then the RDF for the distribution of species B around
                 atoms of species A
      atoms      if not None, a list of atom IDs starting with 1; the RDF
                 will then only be calculated for these select atoms
      cores      Number of cores for parallelization over frames; only
                 reasonable if a trajectory with multiple frames is processed
      normalize  If True, normalize by average atomic density.
      scattering_factors  If True, the RDF will be normalized by the atomic
                 scattering factors (as approximated by the atomic numers)
                 to approximate an experimental pair distribution function

    Returns:
      Dictionary 'rdf' where the keys are the distances and the values are
      the values of the RDF.

      Print the results, for example, with:

          for r in sorted(rdf.keys()):
              print(r, rdf[r])

    """

    if (abs(first) > struc.nframes) or (
            last is not None and abs(last) > struc.nframes):
        print("frames {} to {} out of {}".format(first, last, struc.nframes))
        raise ArgumentError("Specified frame range is invalid.")

    if atoms is None:
        atoms = range(struc.natoms)
    else:
        # subtract 1 as the atom indices start with 1 not 0
        atoms = [i-1 for i in atoms]

    last = struc.nframes if last is None else last
    first -= 1

    dr = float(length/nbins)
    hist = np.zeros(nbins)

    hist_for_frame = functools.partial(_hist_for_frame, struc, nbins,
                                       length, dr, species, atoms,
                                       scattering_factors)

    if cores == 1:
        # hist = np.sum(list(map(hist_for_frame, range(first, last)), axis=0))
        hist = np.sum(list(map(hist_for_frame, range(first, last))), axis=0)
    else:
        pool = mp.Pool(processes=cores)
        hist = np.sum(pool.map(hist_for_frame, range(first, last)), axis=0)

    # divide by squared average atomic scattering factor
    if scattering_factors:
        if species is None:
            # b_avg = np.mean([atomic_number[s] for s in struc.types])
            b_avg = np.mean([atomic_number[struc.types[i]] for i in atoms])
        else:
            # b_avg = np.mean([atomic_number[s]
            #                  for s in struc.types if s in species])
            b_avg = np.mean([atomic_number[struc.types[i]]
                             for i in atoms if struc.types[i] in species])
        hist /= (b_avg*b_avg)

    rdf = {}
    # normalization by average density
    if normalize:
        if species is None:
            # consider all atoms
            rho = struc.natoms/struc.cellvolume()
        else:
            # consider only selected atoms
            N = np.sum(np.where(np.array(struc.types) == species[1], 1, 0))
            rho = N/struc.cellvolume()
        nframes = (last - first)
        a = 4.0/3.0*np.pi*dr**3
        for i in range(nbins):
            r = i*dr
            # A common approximation for the bin volume is
            #   V_bin = 4.0*np.pi*r**2*dr
            # but we use the exact volume here as it is not time critical
            V_bin = a*((i+1)**3 - i**3)
            N_bin = rho*V_bin
            rdf[r + 0.5*dr] = hist[i]/N_bin/nframes
    else:
        for i in range(nbins):
            r = i*dr
            rdf[r + 0.5*dr] = hist[i]

    return rdf


def _hist_for_frame(struc, nbins, length, dr, species, atoms,
                    scattering_factors, frame):
    frame_hist = np.zeros(nbins)
    avec = None if not struc.pbc else struc.avec[frame]
    nbl = NeighborList(struc.coords[frame], lattice_vectors=avec,
                       cartesian=True, types=struc.types,
                       interaction_range=length)
    N_atoms = 0
    for i in atoms:
        if (species is not None) and (struc.types[i] != species[0]):
            continue
        N_atoms += 1
        Zi = atomic_number[struc.types[i]]
        (nb, dist, T) = nbl.get_neighbors_and_distances(i)
        for j in range(len(nb)):
            if (species is not None) and (
                    struc.types[nb[j]] != species[1]):
                continue
            Zj = atomic_number[struc.types[i]]
            rij = dist[j]
            ibin = int(np.floor(rij/dr))
            if (ibin < nbins):
                if scattering_factors:
                    frame_hist[ibin] += Zi*Zj
                else:
                    frame_hist[ibin] += 1.0
    return frame_hist/N_atoms
