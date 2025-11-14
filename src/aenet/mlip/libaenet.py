"""
Low-level CFFI wrapper for libaenet shared library.

This module provides direct access to the aenet C library functions.
It handles the CFFI interface, type conversions, and error handling.

"""

import os
import numpy as np
import threading
import logging
from typing import Tuple, List, Optional, Dict
from cffi import FFI

from .. import config as cfg

logger = logging.getLogger(__name__)

__author__ = "The aenet developers"
__email__ = "aenet@atomistic.net"


# CFFI setup
ffi = FFI()

# Define C interface from aenet.h
ffi.cdef("""
    void aenet_init(int ntypes, char* atom_types[], int* stat);
    void aenet_final(int* stat);
    void aenet_print_info(void);
    void aenet_load_potential(int type_id, char* filename, int* stat);
    void aenet_load_potential_ascii(int type_id, char* filename, int* stat);
    _Bool aenet_all_loaded(void);

    double aenet_free_atom_energy(int type_id);

    void aenet_atomic_energy(double coo_i[3], int type_i, int n_j,
                             double coo_j[], int type_j[], double* E_i,
                             int* stat);

    void aenet_atomic_energy_and_forces(double coo_i[3], int type_i,
                                        int index_i, int n_j, double coo_j[],
                                        int type_j[], int index_j[],
                                        int natoms, double* E_i, double F[],
                                        int* stat);

    void aenet_convert_atom_types(int ntypes_in, char* atom_types[],
                                  int natoms_in, int type_id_in[],
                                  int type_id_out[], int* stat);

    extern int AENET_OK;
    extern int AENET_ERR_INIT;
    extern int AENET_ERR_MALLOC;
    extern int AENET_ERR_IO;
    extern int AENET_ERR_TYPE;
    extern int AENET_TYPELEN;
    extern int AENET_PATHLEN;

    extern int aenet_nsf_max;
    extern int aenet_nnb_max;
    extern double aenet_Rc_min;
    extern double aenet_Rc_max;
""")

# Global library handle
_lib = None
_initialized = False
_atom_types_map = {}


class SessionHandle:
    """Lightweight handle for a libaenet session."""

    def __init__(self, manager: 'LibAenetSessionManager',
                 config: tuple, generation: int):
        self.manager = manager
        self.config = config
        self.generation = generation
        self._released = False

    def release(self):
        """Release this session handle."""
        if not self._released:
            self.manager.release_session(self.config, self.generation)
            self._released = True

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class LibAenetSessionManager:
    """
    Manages global libaenet state with automatic reinitialization.

    This class handles the singleton pattern required by libaenet's global
    state, allowing multiple Python objects to share a single initialized
    library instance. When configuration changes, it automatically finalizes
    the old state and reinitializes with the new configuration.

    Thread-safe for initialization and finalization operations. Read-only
    operations (prediction) after initialization can run concurrently.
    """

    def __init__(self):
        self._lock = threading.RLock()  # Serialize init/finalize
        # (atom_types_tuple, potential_paths_tuple, format)
        self._current_config = None
        self._ref_count = 0  # Active sessions using current config
        self._initialized = False
        self._generation = 0  # Increment on finalize to invalidate old handles

    def acquire_session(
        self,
        atom_types: List[str],
        potential_paths: Dict[str, str],
        potential_format: Optional[str] = None
    ) -> SessionHandle:
        """
        Get or create a session for the given configuration.

        If a session with the same configuration is already active, returns
        a new handle to the existing session. If the configuration differs,
        automatically finalizes the old session and initializes a new one.

        Parameters
        ----------
        atom_types : List[str]
            List of element symbols
        potential_paths : Dict[str, str]
            Mapping from element symbols to potential file paths
        potential_format : str, optional
            Format of potential files ('ascii' or None for binary)

        Returns
        -------
        SessionHandle
            Handle to the session that must be released when done
        """
        with self._lock:
            # Normalize configuration for comparison
            config = (
                tuple(sorted(atom_types)),
                tuple(sorted(potential_paths.items())),
                potential_format
            )

            # Same config: increment reference
            if self._initialized and config == self._current_config:
                self._ref_count += 1
                logger.debug(
                    f"Reusing existing session (ref_count={self._ref_count})")
                return SessionHandle(self, config, self._generation)

            # Different config: finalize old, init new
            if self._initialized:
                logger.info(
                    f"Configuration changed, reinitializing libaenet "
                    f"(old types: {self._current_config[0]}, "
                    f"new types: {config[0]})"
                )
                self._finalize_impl()
                self._initialized = False
                self._generation += 1  # Invalidate any outstanding handles

            # Initialize with new config
            logger.debug(f"Initializing libaenet with types: {atom_types}")
            self._initialize_impl(list(atom_types))

            try:
                for atom_type, path in potential_paths.items():
                    load_potential(atom_type, path, format=potential_format)

                if not all_loaded():
                    raise RuntimeError("Failed to load all potentials")
            except Exception as exc:
                logger.error(
                    "Failed to initialize libaenet session; cleaning up.",
                    exc_info=exc
                )
                self._finalize_impl()
                self._initialized = False
                self._current_config = None
                self._ref_count = 0
                self._generation += 1  # Invalidate any outstanding handles
                raise

            self._current_config = config
            self._initialized = True
            self._ref_count = 1

            return SessionHandle(self, config, self._generation)

    def release_session(self, config: tuple, generation: int):
        """
        Release a session handle.

        Decrements the reference count. When it reaches zero, finalizes
        the library to free resources.

        Parameters
        ----------
        config : tuple
            Configuration tuple from the SessionHandle
        generation : int
            Generation stamp from the SessionHandle
        """
        with self._lock:
            # Ignore releases from old generations
            if generation != self._generation:
                logger.debug("Ignoring release from old generation "
                             f"{generation} (current: {self._generation})")
                return

            if config != self._current_config:
                # Already reinitialized with different config
                logger.debug("Session already superseded by new configuration")
                return

            self._ref_count -= 1
            logger.debug(f"Released session (ref_count={self._ref_count})")

            if self._ref_count == 0:
                logger.debug("No more active sessions, finalizing libaenet")
                self._finalize_impl()
                self._initialized = False
                self._current_config = None
                self._generation += 1  # Invalidate any outstanding handles

    def is_active(self) -> bool:
        """Check if a session is currently active."""
        with self._lock:
            return self._initialized and self._ref_count > 0

    def _initialize_impl(self, atom_types: List[str]):
        """
        Internal initialization that bypasses the already-initialized check.
        """
        global _initialized, _atom_types_map

        lib = _get_library()

        # Create C array of atom type strings
        ntypes = len(atom_types)
        c_atom_types = [ffi.new("char[]", typ.encode('utf-8'))
                        for typ in atom_types]
        c_atom_types_array = ffi.new("char*[]", c_atom_types)

        # Initialize
        stat = ffi.new("int*")
        lib.aenet_init(ntypes, c_atom_types_array, stat)
        _check_status(stat[0], "aenet_init")

        # Store type mapping (symbol -> type_id)
        _atom_types_map = {typ: i + 1 for i, typ in enumerate(atom_types)}
        _initialized = True

    def _finalize_impl(self):
        """Internal finalization that handles cleanup."""
        global _initialized, _atom_types_map

        if not _initialized:
            return

        lib = _get_library()
        stat = ffi.new("int*")
        lib.aenet_final(stat)
        _check_status(stat[0], "aenet_final")

        _initialized = False
        _atom_types_map = {}

    def force_cleanup(self):
        """Forcefully finalize and reset the session manager state."""
        with self._lock:
            if self._initialized:
                self._finalize_impl()
            self._initialized = False
            self._current_config = None
            self._ref_count = 0
            self._generation += 1  # Invalidate any outstanding handles


# Module-level singleton
_session_manager = LibAenetSessionManager()


def cleanup_sessions():
    """Forcefully finalize any active libaenet sessions."""
    _session_manager.force_cleanup()


def _get_library():
    """Get the loaded aenet library handle."""
    global _lib
    if _lib is None:
        aenet_paths = cfg.read('aenet')
        lib_path = aenet_paths.get('aenet_lib_path')

        if lib_path is None:
            raise RuntimeError(
                "aenet library path not configured. "
                "Run 'aenet config' to set aenet_lib_path."
            )

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"aenet library not found at: {lib_path}"
            )

        _lib = ffi.dlopen(lib_path)

    return _lib


class AenetError(Exception):
    """Exception raised for aenet library errors."""

    def __init__(self, message: str, status: int):
        self.status = status
        super().__init__(message)


def _check_status(stat: int, operation: str = "aenet operation"):
    """Check status code and raise exception if error occurred."""
    lib = _get_library()

    if stat == lib.AENET_OK:
        return

    error_messages = {
        lib.AENET_ERR_INIT: "Initialization error",
        lib.AENET_ERR_MALLOC: "Memory allocation error",
        lib.AENET_ERR_IO: "I/O error",
        lib.AENET_ERR_TYPE: "Type error",
    }

    error_msg = error_messages.get(stat, f"Unknown error (code {stat})")
    raise AenetError(f"{operation} failed: {error_msg}", stat)


def initialize(atom_types: List[str]):
    """
    Initialize the aenet library with atom types.

    Parameters
    ----------
    atom_types : List[str]
        List of element symbols (e.g., ['Ti', 'O'])

    Raises
    ------
    AenetError
        If initialization fails
    """
    global _initialized, _atom_types_map

    if _initialized:
        raise RuntimeError(
            "aenet library already initialized. Call finalize() first."
        )

    lib = _get_library()

    # Create C array of atom type strings
    ntypes = len(atom_types)
    c_atom_types = [ffi.new("char[]", typ.encode('utf-8'))
                    for typ in atom_types]
    c_atom_types_array = ffi.new("char*[]", c_atom_types)

    # Initialize
    stat = ffi.new("int*")
    lib.aenet_init(ntypes, c_atom_types_array, stat)
    _check_status(stat[0], "aenet_init")

    # Store type mapping (symbol -> type_id)
    # Note: aenet type IDs are 1-based (1..ntypes)
    _atom_types_map = {typ: i + 1 for i, typ in enumerate(atom_types)}
    _initialized = True


def finalize():
    """
    Finalize and cleanup the aenet library.

    Raises
    ------
    AenetError
        If finalization fails
    """
    global _initialized, _atom_types_map

    if not _initialized:
        return

    lib = _get_library()
    stat = ffi.new("int*")
    lib.aenet_final(stat)
    _check_status(stat[0], "aenet_final")

    _initialized = False
    _atom_types_map = {}


def load_potential(atom_type: str, filename: str,
                   format: Optional[str] = None):
    """
    Load a potential file for a specific atom type.

    Parameters
    ----------
    atom_type : str
        Element symbol (e.g., 'Ti')
    filename : str
        Path to the potential file (.nn or .nn.ascii)
    format : str, optional
        One of {'ascii', 'binary'} (case-insensitive). If None (default),
        the format is auto-detected from the filename extension: files
        ending with '.ascii' (or '.nn.ascii') are treated as ASCII, others
        as binary.

    Raises
    ------
    RuntimeError
        If library not initialized
    ValueError
        If atom type unknown or format string invalid
    AenetError
        If loading fails
    """
    if not _initialized:
        raise RuntimeError(
            "aenet library not initialized. Call initialize() first."
        )

    if atom_type not in _atom_types_map:
        raise ValueError(
            f"Unknown atom type '{atom_type}'. "
            f"Available types: {list(_atom_types_map.keys())}"
        )

    lib = _get_library()
    type_id = _atom_types_map[atom_type]

    c_filename = ffi.new("char[]", filename.encode('utf-8'))
    stat = ffi.new("int*")

    # Determine format
    fmt = (format.lower() if isinstance(format, str) else None)
    if fmt is None:
        is_ascii = (
            filename.endswith('.ascii')
            or filename.endswith('.nn.ascii')
        )
    else:
        if fmt in ('ascii', 'text'):
            is_ascii = True
        elif fmt in ('bin', 'binary'):
            is_ascii = False
        else:
            raise ValueError(f"Unknown potential format: {format!r}")

    # Call appropriate loader
    if is_ascii and hasattr(lib, 'aenet_load_potential_ascii'):
        lib.aenet_load_potential_ascii(type_id, c_filename, stat)
        op_name = f"load_potential_ascii for {atom_type}"
    else:
        lib.aenet_load_potential(type_id, c_filename, stat)
        op_name = f"load_potential for {atom_type}"

    try:
        _check_status(stat[0], op_name)
    except AenetError as e:
        # Provide hint if likely wrong format selected
        if e.status == lib.AENET_ERR_TYPE:
            hint = (
                " Hint: try potential_format='ascii' or ensure the filename "
                "ends with '.ascii'."
                if not is_ascii
                else " Hint: try binary format for this file."
            )
            raise AenetError(
                f"{op_name} failed: Type error.{hint}",
                e.status
            )
        raise


def all_loaded() -> bool:
    """
    Check if all potentials have been loaded.

    Returns
    -------
    bool
        True if all potentials are loaded

    Raises
    ------
    RuntimeError
        If library not initialized
    """
    if not _initialized:
        raise RuntimeError(
            "aenet library not initialized. Call initialize() first."
        )

    lib = _get_library()
    return bool(lib.aenet_all_loaded())


def free_atom_energy(atom_type: str) -> float:
    """
    Get the free atom energy for a specific atom type.

    Parameters
    ----------
    atom_type : str
        Element symbol

    Returns
    -------
    float
        Free atom energy in eV

    Raises
    ------
    RuntimeError
        If library not initialized
    ValueError
        If atom type unknown
    """
    if not _initialized:
        raise RuntimeError(
            "aenet library not initialized. Call initialize() first."
        )

    if atom_type not in _atom_types_map:
        raise ValueError(
            f"Unknown atom type '{atom_type}'. "
            f"Available types: {list(_atom_types_map.keys())}"
        )

    lib = _get_library()
    type_id = _atom_types_map[atom_type]
    return lib.aenet_free_atom_energy(type_id)


def atomic_energy_and_forces(
    coo_i: np.ndarray,
    type_i: int,
    index_i: int,
    coo_j: np.ndarray,
    type_j: np.ndarray,
    index_j: np.ndarray,
    natoms: int,
    forces: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Calculate atomic energy and forces for a single atom.

    Parameters
    ----------
    coo_i : np.ndarray
        Coordinates of central atom, shape (3,)
    type_i : int
        Type ID of central atom
    index_i : int
        Index of central atom in the structure
    coo_j : np.ndarray
        Coordinates of neighbor atoms, shape (n_neighbors, 3)
    type_j : np.ndarray
        Type IDs of neighbor atoms, shape (n_neighbors,)
    index_j : np.ndarray
        Indices of neighbor atoms, shape (n_neighbors,)
    natoms : int
        Total number of atoms in the structure
    forces : np.ndarray, optional
        Pre-allocated force array, shape (natoms, 3).
        If None, a new array is created.

    Returns
    -------
    energy : float
        Atomic energy contribution in eV
    forces : np.ndarray
        Force contributions, shape (natoms, 3) in eV/Angstrom

    Raises
    ------
    RuntimeError
        If library not initialized
    AenetError
        If calculation fails
    """
    if not _initialized:
        raise RuntimeError(
            "aenet library not initialized. Call initialize() first."
        )

    lib = _get_library()

    # Ensure contiguous arrays with correct dtype
    coo_i = np.ascontiguousarray(coo_i, dtype=np.float64)
    coo_j = np.ascontiguousarray(coo_j.ravel(), dtype=np.float64)
    type_j = np.ascontiguousarray(type_j, dtype=np.int32)
    index_j = np.ascontiguousarray(index_j, dtype=np.int32)

    # Prepare output arrays
    if forces is None:
        forces = np.zeros((natoms, 3), dtype=np.float64)
    else:
        forces = np.ascontiguousarray(forces, dtype=np.float64)

    E_i = ffi.new("double*")
    stat = ffi.new("int*")

    n_j = len(type_j)

    # Call library function
    lib.aenet_atomic_energy_and_forces(
        ffi.cast("double*", ffi.from_buffer(coo_i)),
        type_i,
        index_i,
        n_j,
        ffi.cast("double*", ffi.from_buffer(coo_j)),
        ffi.cast("int*", ffi.from_buffer(type_j)),
        ffi.cast("int*", ffi.from_buffer(index_j)),
        natoms,
        E_i,
        ffi.cast("double*", ffi.from_buffer(forces)),
        stat
    )

    _check_status(stat[0], "atomic_energy_and_forces")

    return E_i[0], forces


def is_initialized() -> bool:
    """Check if the library is initialized."""
    return _initialized


def get_atom_types() -> List[str]:
    """Get the list of initialized atom types."""
    return list(_atom_types_map.keys())


def get_type_id(atom_type: str) -> int:
    """
    Get the type ID for an atom type.

    Parameters
    ----------
    atom_type : str
        Element symbol

    Returns
    -------
    int
        Type ID used by aenet

    Raises
    ------
    ValueError
        If atom type unknown
    """
    if atom_type not in _atom_types_map:
        raise ValueError(
            f"Unknown atom type '{atom_type}'. "
            f"Available types: {list(_atom_types_map.keys())}"
        )
    return _atom_types_map[atom_type]


def get_cutoff_radius() -> Tuple[float, float]:
    """
    Get the minimum and maximum cutoff radii.

    Returns
    -------
    Rc_min : float
        Minimum cutoff radius in Angstrom
    Rc_max : float
        Maximum cutoff radius in Angstrom

    Raises
    ------
    RuntimeError
        If library not initialized or potentials not loaded
    """
    if not _initialized:
        raise RuntimeError(
            "aenet library not initialized. Call initialize() first."
        )

    if not all_loaded():
        raise RuntimeError(
            "Not all potentials loaded. Call load_potential() first."
        )

    lib = _get_library()
    return float(lib.aenet_Rc_min), float(lib.aenet_Rc_max)
