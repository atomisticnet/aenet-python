"""
Geometry-specific utility functions.

"""

import numpy as np
import sys
import itertools

from ..exceptions import ArgumentError, InternalError

__author__ = "Alexander Urban"
__date__ = "2013-03-25"

EPS = np.finfo(np.double).eps


def add_redundant_atoms(coords, types, mult=[2, 1, 1]):
    """
    Add periodic images of atoms to the list of coordinates.  This
    operation can be useful/required before transforming the unit cell.

    Arguments:
      coords     list of *fractional* coordinates
      types      list of atomic species of atoms in `coords'
      mult       list of three integers defining the supercell within
                 which the redundant atoms shall exist

    Output:
      tuple (coords_red, types_red) where coords_red is a list of
      redundant atomic coordinates and types_red is a list of the
      corresponding atomic types.
      `coords_red' and `types_red' are NumPy ndarrays.
    """

    coords_red = []
    types_red = []
    for ix in range(mult[0]):
        for iy in range(mult[1]):
            for iz in range(mult[2]):
                for iat in range(np.size(coords, axis=0)):
                    c = np.array(coords[iat]) + np.array([ix, iy, iz])
                    coords_red.append(c)
                    types_red.append(types[iat])

    coords_red = np.array(coords_red)
    types_red = np.array(types_red)

    return (coords_red, types_red)


def del_redundant_atoms(coords, types, dr=1.0e-3):
    """
    Delete redundant periodic images from a list of coordinates.

    Arguments:
      coords     list of *fractional* coordinates
      types      list of atomic species of atoms in `coords'
      dr         precision for comparison of coordinates

    Output:
      tuple (coords_uniq, types_uniq) where coords_uniq is a list of
      periodically unique atomic coordinates and types_uniq is a list of
      the corresponding atomic types.
      `coords_uniq' and `types_uniq' are NumPy ndarrays.
    """

    coords_uniq = []
    types_uniq = []
    for i in range(len(coords)):
        c1 = coords[i]
        add = True
        for c2 in coords_uniq:
            vec = c2 - c1
            if (np.linalg.norm(vec) < dr):
                add = False
                break
        if (add):
            coords_uniq.append(c1)
            types_uniq.append(types[i])
    coords_uniq = np.array(coords_uniq)
    types_uniq = np.array(types_uniq)
    return (coords_uniq, types_uniq)


def cellmatrix_from_params(a, b, c, alpha, beta, gamma, rad=False):
    """
    Return matrix of lattice vectors for a given set of cell paramters.

    Arguments:
      a, b, c             lattice constants = lengths of lattice vectors
      alpha, beta, gamma  cell angles = angles between lattice vectors
      rad                 if True, angles are in radiants (not degrees)

    Returns:
      3x3 ndarray A, with A[i] = (i+1)-th lattice vector; i = 0, 1, 2
    """

    if not rad:
        alpha = alpha/180.0*np.pi
        beta = beta/180.0*np.pi
        gamma = gamma/180.0*np.pi

    # a*b = a1*b1 = |a|*|b|*cos(gamma)
    # a*c = a1*c1 = |a|*|c|*cos(beta)
    # b*c = b1*c1 + b2*c2 = |b|*|c|*cos(alpha)

    b1 = b*np.cos(gamma)
    b2 = np.sqrt(b*b - b1*b1)
    c1 = c*np.cos(beta)
    c2 = (b*c*np.cos(alpha) - b1*c1)/b2
    c3 = np.sqrt(c*c - c1*c1 - c2*c2)

    avec = np.array([[a, 0.0, 0.0],
                     [b1, b2, 0.0],
                     [c1, c2, c3]])

    return avec


def center_of_mass(coords, masses=None, avec=None, translations=False):
    """
    Calculates the center of mass (COM) of a list of weighted
    coordinates.  With periodic boundary conditions there is no well
    defined COM, and the function will return the one choice that is
    closest (by distance) to all points.

    Arguments:
      coords[i][j]   j-th Cartesian coordinate of i-th point
      masses[i]      mass/weight of i-th point
      avec[i][j]     j-th component of the i-th lattice vector
      translations   if True, the T vectors of the sites will be returned

    Returns:
      tuple COM = (com_x, com_y, com_z) of Cartesian coordinates
      or, if translation is true, (COM, T) will be returned, where
      T is a list of translation vectors
    """

    N = len(coords)
    if masses is None:
        masses = np.ones(N)

    M = np.sum(masses)
    com0 = np.sum((masses*coords.T), 1)/M
    T_opt = N*[np.array([0, 0, 0])]

    if avec is None:
        com = com0
        translations = False
    else:
        # Brute force loop over 27 translations star of translation
        # vectors.  This can get lengthy for sites with large
        # coordination numbers.
        T_star = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    T_star.append(np.array([i, j, k]))
        trials = [[(coords[0], [0.0, 0.0, 0.0])]]
        for i in range(1, N):
            coo = []
            for T in T_star:
                coo.append((coords[i] + np.dot(T, avec), T))
            trials.append(coo)
        d_min = None
        for combi in itertools.product(*trials):
            coords2 = np.array([c for (c, T) in combi])
            com_test = np.sum((masses*coords2.T), 1)/M
            d_mean = np.mean([np.linalg.norm(c - com_test) for c in coords2])
            if (d_min is None) or (d_mean < d_min - 0.01):
                d_min = d_mean
                T_opt = np.array([T for (c, T) in combi], dtype=int)
                com = com_test.copy()
        com_frac = np.dot(com, np.linalg.inv(avec))
        for i in range(3):
            while (com_frac[i] >= 1.0):
                com_frac[i] -= 1.0
                T_opt[:, i] -= 1
            while (com_frac[i] < 0.0):
                com_frac[i] += 1.0
                T_opt[:, i] += 1
        com = np.dot(com_frac, avec)

    if translations:
        return (com, T_opt)
    else:
        return com


def standard_cell(avec, angles=False, areas=False):
    """
    Rotate lattive vectors to standard cell.

    The standard lattice vector matrix has the triangular form:

       a = (a1,  0,  0)^T
       b = (b1, b2,  0)^T
       c = (c1, c2, c3)^T

    Arguments:
      avec[i][j]  (ndarray) j-th component of the i-th lattice vector
      angles (bool)         return also lattice constants and angles
      areas (bool)          return also areas spanned by lattice vectors

    Returns:

      | angles | areas | returned                                            |
      |----------------------------------------------------------------------|
      | False  | False | avec_new                                            |
      | True   | False | (avec_new, a, b, c, alpha, beta, gamma)             |
      | True   | True  | (avec_new, a, b, c, alpha, beta, gamma, ab, ac, bc) |
      | False  | True  | (avec_new, ab, ac, bc)                              |

      avec_new (ndarray)   rotated lattice basis
      a, b, c (int)        lengths of the lattice vectors
      alpha, beta, gamma   angles
      ab, ac, bc           areas spanned by the vectors

    """

    # new cell dimensions
    a = np.linalg.norm(avec[0])
    b = np.linalg.norm(avec[1])
    c = np.linalg.norm(avec[2])

    # rotated lattice vectors
    ab = np.dot(avec[1], avec[0])
    ac = np.dot(avec[0], avec[2])
    bc = np.dot(avec[1], avec[2])
    a1 = a
    b1 = ab/a
    b2 = np.sqrt(b*b - b1*b1)
    c1 = ac/a
    c2 = (bc - b1*c1)/b2
    c3 = np.sqrt(c*c - c1*c1 - c2*c2)

    avec_new = np.array([[a1, 0.0, 0.0],
                         [b1, b2,  0.0],
                         [c1, c2,  c3]])

    if angles:
        # cell angles
        alpha = np.arccos(bc/(b*c))/np.pi*180
        beta = np.arccos(ac/(a*c))/np.pi*180
        gamma = np.arccos(ab/(a*b))/np.pi*180

    if areas:
        # areas of the parallelograms spanned by lattice vectors
        abarea = abs(a1*b2)
        acarea = np.linalg.norm(np.cross([a1, 0, 0], [c1, c2, c3]))
        bcarea = np.linalg.norm(np.cross([b1, b2, 0], [c1, c2, c3]))

    if angles and areas:
        return (avec_new, a, b, c, alpha, beta, gamma, abarea, acarea, bcarea)
    elif angles:
        return (avec_new, a, b, c, alpha, beta, gamma)
    elif areas:
        return (avec_new, abarea, acarea, bcarea)
    else:
        return avec_new


def transform_cell(avec, coords, types, T, sort=2):
    """
    Transform the periodic cell described by lattice matrix AVEC by
    applying the transformation matrix T.

    Arguments:
      avec[i][j]    j-th component of the i-th lattice vector
      coords        list of *fractional* atomic coordinates in the
                    coordinate system defined by `avec'
      types         list of atomic types corresponding to `coords'
      T             3x3 matrix that describes the transformation
      sort          index for sorting the final coordinates (2==z); set
                    to None for no sorting; in addition, coordinates
                    will always be sorted by atom type

    Output:
      tuple (avec_T, coords_T, types_T)

    Example transformation:

      For the input cell avec = [a, b, c] the transformation matrix

         T = np.array([[1, -1,  0],
                       [1,  0, -1],
                       [1,  1,  1]])

      will result in a new coordinate system avec' = [a', b', c'] with

         avec' = T . avec

         => a' = a - b
         => b' = a     - c
         => c' = a + b + c   .
    """

    T = np.array(T)
    U = np.linalg.inv(T)
    avec_T = np.dot(T, avec)

    # Compare cell volumina for error catching and atom numbers.
    V = np.linalg.det(avec)
    if (V < 0.0):
        sys.stderr.write(
            "\n Warning: the original lattice basis is left-handed !\n\n")
        V = abs(V)
    V_T = np.linalg.det(avec_T)
    if (V_T < 0.0):
        sys.stderr.write(
            "\n Warning: the new lattice basis is left-handed !\n\n")
        V_T = abs(V_T)
    s = V_T/V
    natoms = len(coords)
    natoms_T = np.round(s*natoms)

    # Stop, if transformation would yield fractional atom count.
    if (abs(s*natoms - natoms_T) > 0.01):
        sys.stderr.write("V = {}\nV_T = {}\n".format(V, V_T))
        sys.stderr.write("V/V_T = {}\nN_atoms = {}\n".format(V_T/V, s*natoms))
        raise ArgumentError("Invalid transformation in `transform_cell'.")

    # The following block of code adds redundant periodic images of the
    # atoms in coords to the list of coordinates, and transforms the
    # coordinates to the new basis.  If the resulting number of atoms is
    # lower than expected, more redundant coordinates are needed.  The
    # maximum numbe of redundant coordinates is set to (s+3)^3, where s
    # is the scaling factor that relates the new cell to the old one..
    N = int(np.ceil(s)) + 3
    n = int(np.ceil(s**(1./3.)))
    mult = [n, n, n]
    natoms_new = 0
    while (natoms_new != natoms_T) and (mult[0] <= N):
        while (natoms_new != natoms_T) and (mult[1] <= mult[0]):
            while (natoms_new != natoms_T) and (mult[2] <= mult[1]):
                (coords_T, types_T
                 ) = add_redundant_atoms(coords, types, mult=mult)
                coords_T = np.dot(coords_T, U)
                coords_T = wrap_pbc(coords_T)
                (coords_T, types_T
                 ) = del_redundant_atoms(coords_T, types_T, dr=0.05)
                natoms_new = len(coords_T)
                mult[2] += 1
            mult[1] += 1
        mult[0] += 1

    if (natoms_new != natoms_T):
        sys.stderr.write("{} atoms found, should be {}\n".format(
            natoms_new, natoms_T))
        raise InternalError(
            "Basis transformation failed in `transform_cell()'.")

    # Sort atoms by requested coordinate (if any).
    if sort is not None:
        sort = max(min(sort, 2), 0)
        idx = np.argsort(coords_T[:, sort])
        types_T = types_T[idx]
        coords_T = coords_T[idx]

    # Always sort atoms by atom type.
    idx = np.argsort(types_T)
    types_T = types_T[idx]
    coords_T = coords_T[idx]

    return (avec_T, coords_T, types_T)


def rotation_matrix(axis, angle):
    """
    Euler-Rodrigues rotation matrix, R = Rz . Ry . Rx

    Returns the rotation matrix for the rotation about arbitrary axis by
    given angle.

    Arguments:
      axis  (list)   3-d vector in direction of the rotation axis
      angle (float)  angle of the rotation in radians

    """

    axis = axis/np.linalg.norm(axis)
    a = np.cos(angle/2)
    b, c, d = -axis*np.sin(angle/2)

    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


def rotate(array, vectors=None, axis=None, angle=None, degrees=False):
    """
    Rotate coordinates in an input array. The axis and angle of the
    rotation can either be directly specified, or in terms of two
    reference vectors.  In the second case, the axis is chosen
    perpendicular to the two given vectors when not otherwise specified,
    and the angle of rotation is the angle between the two vectors.

    Arguments:
      array    single vector or matrix (list of vectors) to be rotated
      vectors  list of two reference vectors v1 and v2
      axis     vector in the direction of the rotation axis
      angle    angle of the rotation
      degrees  If True, the rotation angle is specified in degrees; default
               is radians

    Returns:
      a rotated copy of the input array

    Example 1:
      array = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
      axis = [1.0, 1.0, 1.0]
      angle = 45.0
      array_rot = rotate(array, axis_and_angle=(axis, angle), degrees=True)

    Example 2:
      array = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
      v1 = array[0]
      v2 = [1.0, 0.0, 0.0]
      array_rot = rotate(array, vectors=(v1, v2))

    """

    if vectors is None and axis is None and angle is None:
        raise ArgumentError("No rotation defined. "
                            "Vectors or axis/angle required.")

    if bool(vectors is not None) == bool(angle is not None):
        raise ArgumentError("Check angle specification. "
                            "Use either vectors or explicit angle.")

    if vectors is None and axis is None:
        raise ArgumentError("No rotation axis specified. "
                            "Use either vectors or explicit axis.")

    if vectors is not None:
        v1 = vectors[0]
        v2 = vectors[1]

    if axis is None:  # if axis is None vectors must be present
        axis = np.cross(v1, v2)
        axis_is_perpendicular = True
    else:
        axis_is_perpendicular = False

    if angle is None:  # if angle is None vectors must be present
        if axis_is_perpendicular:
            p1 = v1
            p2 = v2
        else:
            # determine vector components p1 and p2 of v1 and v2 that are
            # perpendicular to the axis of rotation
            axis_dir = axis/np.linalg.norm(axis)
            p1 = v1 - np.dot(v1, axis_dir)*axis_dir
            p2 = v2 - np.dot(v2, axis_dir)*axis_dir
        len_p1 = np.linalg.norm(p1)
        len_p2 = np.linalg.norm(p2)
        angle = -np.arccos(np.dot(p1, p2)/(len_p1*len_p2))
    elif degrees:
        angle *= np.pi/180.0

    if (np.abs(angle) > np.sqrt(EPS)):
        R = rotation_matrix(axis, angle)
        array_rot = np.dot(array, R.T)
    else:
        array_rot = array

    return array_rot


def symmetry_equivalent_atoms(coords, types, symmetry_operations, tol=1.0e-4):
    """
    Generate symmetrically equivalent atoms based on an assymetric unit
    and a list of symmetry operations.

    Arguments:
      coords (list)    Fractional atomic coordinates of the assymetric unit
      types (list)     Atomic species (str) for each atom in the AU
      symmetry_operations (list of arrays)
                       List of tuples (S, T) where S is a 3x3 matrix
                       representations of the symmetry operation and T is
                       a translation vector.
      tol (float)      Minimum distance (in reciprocal units) between atoms
                       to count as distinct.

    Returns:
      tuple (equivalent_coords, equivalent_types)

    """

    ecoords = []
    etypes = []
    for i, coo in enumerate(coords):
        for S, T in symmetry_operations:
            R = wrap_pbc(np.dot(S, coo) + np.array(T))
            # R = np.dot(S, coo) + np.array(T)
            keep = True
            for R2 in ecoords:
                dist = np.linalg.norm(R2 - R)
                if dist < tol:
                    keep = False
                    break
            if keep:
                ecoords.append(R.copy())
                etypes.append(types[i])

    return (np.array(ecoords), etypes)


def strain_orthorhombic(avec, e_xx):
    """
    Apply volume conserving orthorhombic strain to the input unit cell.

      A_new = A . (I - E)

      with strain tensor

      E = [[ e_xx,    0,    0 ],
           [    0, e_yy,    0 ],
           [    0,    0, e_zz ]

      e_yy = -e_xx  and  e_zz = e_xx**2/(1.0 - e_xx**2)

    This deformation can be used to calculate the C11-C12 elastic
    modulus of cubic structures.

    Arguments:
      avec (2d array)   lattice vectors of the input cell
      e_xx (float)      first component of the strain tensor

    Returns:
      avec_new (2d array)   new deformed lattice vectors
    """

    J = np.identity(3)

    # volume conserving orthorhombic strain
    # --> (C11 - C12) modulus
    e1 = e_xx
    e2 = -e_xx
    e3 = e_xx**2/(1.0 - e_xx**2)
    e4 = e5 = e6 = 0

    # strain tensor
    E = np.array([[e1,     0.5*e6, 0.5*e5],
                  [0.5*e6,     e2, 0.5*e4],
                  [0.5*e5, 0.5*e4,     e3]])

    avec_new = avec.dot(J + E)

    return avec_new


def strain_monoclinic(avec, gamma_xy):
    """
    Apply volume conserving monoclinic strain to the input unit cell.

      A_new = A . (I - E)

      with strain tensor

      E = [[          0, gamma_xy/2,    0 ],
           [ gamma_xy/2,          0,    0 ],
           [          0,          0, e_zz ]

      e_zz = gamma_xy**2/(4.0 - gamma_xy**2)

    This deformation can be used to calculate the C44 elastic modulus
    of cubic structures.

    Arguments:
      avec (2d array)   lattice vectors of the input cell
      gamma_xy (float)  engineering shear strain gamma_xy

    Returns:
      avec_new (2d array)   new deformed lattice vectors
    """

    J = np.identity(3)

    # volume conserving monoclinic strain
    # --> C44 modulus
    e6 = gamma_xy
    e3 = gamma_xy**2/(4.0 - gamma_xy**2)
    e1 = e2 = e4 = e5 = 0

    # strain tensor
    E = np.array([[e1,     0.5*e6, 0.5*e5],
                  [0.5*e6,     e2, 0.5*e4],
                  [0.5*e5, 0.5*e4,     e3]])

    avec_new = avec.dot(J + E)

    return avec_new


def wrap_pbc(coo):
    """
    Wrap fractional coordinates back into the unit cell.

    Arguments:
      coo         3-vector of fractional coordinates
    or
      coo[i][j]   j-th component of the fractional coordinates
                  of the i-th atom.

    Returns:
      0.0 <= coo[i][j] < 1.0  for all i, j
    """

    try:
        # if coo[0] is iterable, coo is a list of coordinates vectors;
        # otherwise it is just a single vector with 3 components
        iter(coo[0])
    except TypeError:
        for i in range(3):
            while (coo[i] < 0.0):
                coo[i] += 1.0
            while (coo[i] >= (1.0 - 1.0e5*EPS)):
                coo[i] -= 1.0
    else:
        for vec in coo:
            for i in range(3):
                while (vec[i] < 0.0):
                    vec[i] += 1.0
                while (vec[i] >= (1.0 - 1.0e5*EPS)):
                    vec[i] -= 1.0

    return coo


# ==============================================================================
# Vector set diversity analysis
# ==============================================================================

def standardize(vecs):
    """
    Standardize vectors to zero mean and unit variance.

    This function centers each feature (column) to have mean zero and
    scales it to have unit variance. This is often a useful preprocessing
    step before computing diversity metrics.

    Parameters
    ----------
    vecs : array_like of shape (N, d)
        Input vectors where rows are samples and columns are features.

    Returns
    -------
    vecs_std : ndarray of shape (N, d)
        Standardized vectors with zero mean and unit variance per feature.

    Notes
    -----
    Uses sample standard deviation (ddof=1). Features with zero variance
    are left unchanged (divided by 1 instead of 0 to avoid division errors).

    Examples
    --------
    >>> vecs = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    >>> vecs_std = standardize(vecs)
    >>> np.allclose(vecs_std.mean(axis=0), 0.0)
    True
    >>> np.allclose(vecs_std.std(axis=0, ddof=1), 1.0)
    True
    """
    vecs = np.asarray(vecs)
    mu = vecs.mean(axis=0)
    sigma = vecs.std(axis=0, ddof=1)  # Use sample std (ddof=1)

    # Handle zero variance columns to avoid division by zero
    sigma = np.where(sigma < 1e-10, 1.0, sigma)

    vecs_std = (vecs - mu) / sigma
    return vecs_std


def entropy_from_cov_regularized(vecs, alpha=1e-3):
    """
    Compute log-determinant of regularized covariance matrix.

    This function efficiently computes the log-determinant of a
    regularized covariance matrix using the Matrix Determinant Lemma
    when N < d (more efficient than computing the full d×d determinant).

    The regularized covariance is Σ_ε = Σ + ε·I where ε is adaptively
    scaled based on the average variance in the data.

    Parameters
    ----------
    vecs : array_like of shape (N, d)
        Input vectors where rows are samples (N) and columns are
        features (d).
    alpha : float, optional
        Regularization parameter as a fraction of average variance
        (default: 1e-3). The actual regularization is ε = α · avg_var.

    Returns
    -------
    log_det : float
        Log-determinant of the regularized covariance matrix.
        Returns -inf if the matrix is not positive definite.

    Notes
    -----
    The implementation uses the identity:
        log det(Σ_ε) = (d - N)·log(ε) + log det(M)
    where M = ε·I_N + (1/(N-1))·X·X^T is an N×N matrix.

    This is efficient when N << d (e.g., N=10 samples in d=3000-dimensional
    space), as we only need to compute the determinant of an N×N matrix
    rather than a d×d matrix.

    The log-determinant is proportional to the differential entropy of a
    Gaussian distribution with covariance Σ_ε, making it a natural measure
    of diversity and information content in the vector set.

    References
    ----------
    Matrix Determinant Lemma:
    https://en.wikipedia.org/wiki/Matrix_determinant_lemma

    Examples
    --------
    >>> vecs = np.random.randn(10, 100)
    >>> log_det = entropy_from_cov_regularized(vecs)
    >>> isinstance(log_det, float)
    True
    """
    vecs = np.asarray(vecs)
    N, d = vecs.shape

    if N <= 1:
        raise ValueError(
            "Need at least 2 samples to compute covariance entropy"
        )

    # Center the data
    X = vecs - vecs.mean(axis=0, keepdims=True)

    # Compute sample covariance trace for adaptive regularization scaling
    cov_trace = np.sum(X ** 2) / (N - 1)
    avg_var = cov_trace / d
    eps = alpha * avg_var

    # Construct N×N matrix: M = ε·I_N + (1/(N-1))·X·X^T
    # This is more efficient than computing d×d covariance when N << d
    XXt = X @ X.T / (N - 1)
    M = XXt + eps * np.eye(N)

    # Compute log determinant using sign and log
    sign, logdet_M = np.linalg.slogdet(M)
    if sign <= 0:
        # Non-positive determinant indicates numerical issues
        import warnings
        warnings.warn(
            "Non-positive determinant encountered in covariance matrix. "
            "Returning -inf.",
            RuntimeWarning
        )
        return -np.inf

    # Apply Matrix Determinant Lemma:
    # log det(Σ_ε) = (d - N)·log(ε) + log det(M)
    logdet_Sigma_eps = (d - N) * np.log(eps) + logdet_M

    return logdet_Sigma_eps


def diversity_metrics(vecs):
    """
    Compute comprehensive diversity metrics for a set of vectors.

    This function quantifies the diversity and information content of a
    vector set using multiple complementary metrics. It is particularly
    useful for analyzing sets of feature vectors from atomic structures
    or displacement patterns in structural sampling.

    Parameters
    ----------
    vecs : array_like of shape (N, d)
        N vectors of dimension d, where rows are samples and columns
        are features.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mean_euclidean_dist' : float
            Mean pairwise Euclidean distance between vectors.
        - 'mean_pearson_corr' : float
            Mean pairwise Pearson correlation coefficient.
        - 'mean_cosine_sim' : float
            Mean pairwise cosine similarity.
        - 'log_det_cov' : float
            Log-determinant of regularized covariance matrix (related
            to differential entropy for Gaussian distributions).
        - 'n_vectors' : int
            Number of vectors in the set.
        - 'dimension' : int
            Dimensionality of each vector.

    Raises
    ------
    ValueError
        If fewer than 2 vectors are provided.

    Notes
    -----
    **Interpretation of metrics:**

    - **Euclidean distance**: Larger values indicate vectors are more
      spread out in space. For fixed-norm vectors, maximizing pairwise
      distances is equivalent to enforcing zero mean.

    - **Pearson correlation**: Measures linear relationships between
      vectors. Values near 0 indicate orthogonal (uncorrelated) vectors,
      which is desirable for diverse sampling.

    - **Cosine similarity**: Measures angular similarity. Values near 0
      indicate orthogonal directions, while ±1 indicates parallel or
      anti-parallel vectors.

    - **Log-det covariance**: Quantifies the "volume" occupied by the
      vector set in feature space. Larger values indicate more diverse,
      less redundant sampling. This is the D-optimality criterion from
      experimental design theory.

    **Applications:**

    - Quantifying diversity of atomic displacement patterns
    - Evaluating feature vector sets from structure featurization
    - Assessing training set coverage in machine learning
    - Optimal experimental design (D-optimal designs)

    References
    ----------
    Pukelsheim, F. (2006). Optimal Design of Experiments. SIAM.

    Examples
    --------
    >>> vecs = np.random.randn(20, 50)
    >>> metrics = diversity_metrics(vecs)
    >>> print(f"Mean distance: {metrics['mean_euclidean_dist']:.3f}")
    >>> print(f"Log-det: {metrics['log_det_cov']:.3f}")

    See Also
    --------
    format_diversity_metrics : Format metrics dictionary as a string.
    entropy_from_cov_regularized : Compute log-det of covariance.
    standardize : Standardize vectors before computing metrics.
    """
    vecs = np.asarray(vecs)
    N, d = vecs.shape

    if N < 2:
        raise ValueError(
            f"Need at least 2 vectors for diversity metrics, got {N}"
        )

    # Compute pairwise metrics
    dists = []
    pearson_corrs = []
    cosine_sims = []

    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = vecs[i], vecs[j]

            # Euclidean distance
            dists.append(np.linalg.norm(v2 - v1))

            # Pearson correlation
            # Use numpy's corrcoef for efficiency
            if d > 1:
                corr_matrix = np.corrcoef(v1, v2)
                pearson_corrs.append(corr_matrix[0, 1])
            else:
                # For 1D, correlation is ±1 or undefined
                pearson_corrs.append(1.0 if np.allclose(v1, v2) else 0.0)

            # Cosine similarity
            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_prod > 1e-10:
                cosine_sims.append(np.dot(v1, v2) / norm_prod)
            else:
                cosine_sims.append(0.0)

    # Entropy from covariance (log-det)
    log_det = entropy_from_cov_regularized(vecs)

    return {
        'mean_euclidean_dist': np.mean(dists),
        'mean_pearson_corr': np.mean(pearson_corrs),
        'mean_cosine_sim': np.mean(cosine_sims),
        'log_det_cov': log_det,
        'n_vectors': N,
        'dimension': d,
    }


def format_diversity_metrics(metrics, header=False):
    """
    Format diversity metrics dictionary as a compact string.

    Parameters
    ----------
    metrics : dict
        Dictionary returned by diversity_metrics().
    header : bool, optional
        If True, return a header string in addition to the values
        (default: False).

    Returns
    -------
    formatted : str
        Formatted string with key metrics (distance, correlation,
        cosine similarity, log-det).

    Examples
    --------
    >>> vecs = np.random.randn(10, 20)
    >>> metrics = diversity_metrics(vecs)
    >>> print(format_diversity_metrics(metrics))
    1.234, 0.056, 0.123, 45.678

    See Also
    --------
    diversity_metrics : Compute diversity metrics for vector sets.
    """
    header_str = (
        "Mean_Euclidean_Dist, "
        "Mean_Pearson_Corr, "
        "Mean_Cosine_Sim, "
        "Log_Det_Cov" + "\n"
    ) if header else ""
    return header_str + (
        f"{metrics['mean_euclidean_dist']:.3f}, "
        f"{metrics['mean_pearson_corr']:.3f}, "
        f"{metrics['mean_cosine_sim']:.3f}, "
        f"{metrics['log_det_cov']:.3f}"
    )
