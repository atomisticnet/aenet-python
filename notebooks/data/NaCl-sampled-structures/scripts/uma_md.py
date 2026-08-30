#!/usr/bin/env python3
"""Run the UMA NPT protocol used to generate the NaCl candidate pool.

Run this script from a directory whose parent is named for the target
temperature, for example ``550K/run``.  The original runs did not record a
random seed, so the tracked XSF archive is the authoritative dataset.
"""

import sys
import time
from pathlib import Path

from temperature_utils import temperature_name_from_parent

timestep_fs = 2.0
equilibration_ps = 20.0
production_ps = 100.0
production_frames = 5000

equilibration_steps = int(equilibration_ps * 1000 / timestep_fs)
production_steps = int(production_ps * 1000 / timestep_fs)
trajectory_interval = production_steps // production_frames

if production_steps % production_frames != 0:
    raise ValueError(
        "production_steps must divide evenly into production_frames"
    )


def main() -> None:
    """Run equilibration and production MD at the directory temperature."""
    from ase import units
    from ase.io import read, write
    from ase.io.trajectory import TrajectoryWriter
    from ase.md import MDLogger
    from ase.md.npt import NPT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    start_time = time.time()

    folder = Path.cwd()
    T_target = int(temperature_name_from_parent(folder))
    temperature_label = f"{T_target}K"

    vasp_files = list(folder.glob("*.vasp"))
    if not vasp_files:
        print(
            "ERROR: No .vasp file found in the current directory.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    elif len(vasp_files) > 1:
        print(
            f"WARNING: Multiple .vasp files found, using {vasp_files[0].name}",
            flush=True,
        )

    vasp_file = vasp_files[0]
    print(f"Using VASP file: {vasp_file.name}", flush=True)

    atoms = read(vasp_file)

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    uma = FAIRChemCalculator(predictor, task_name="omat")
    atoms.calc = uma

    MaxwellBoltzmannDistribution(atoms, temperature_K=T_target)

    dyn = NPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=T_target,
        externalstress=0.0,
        ttime=100 * units.fs,
        pfactor=40000 * units.fs**2,
    )

    log_path = f"NPT_{temperature_label}.log"
    md_logger = MDLogger(
        dyn, atoms, log_path, header=True, stress=True, peratom=False, mode="w"
    )
    dyn.attach(md_logger, interval=100)

    print(
        f"Equilibrating for {equilibration_ps:g} ps ({equilibration_steps} steps)..."
    )
    dyn.run(equilibration_steps)

    traj_path = f"production_{temperature_label}.traj"
    traj_writer = TrajectoryWriter(str(traj_path), mode="w", atoms=atoms)

    # Keep frame 0 as the post-equilibration structure; traj_to_xsf.py skips it.
    traj_writer.write(atoms)
    dyn.attach(traj_writer.write, interval=trajectory_interval)

    print(
        f"Writing {production_frames} production frames over {production_ps:g} ps "
        f"to: {traj_path}"
    )
    dyn.run(production_steps)
    traj_writer.close()

    final_vasp = f"final_{temperature_label}.vasp"
    atoms.wrap()
    write(final_vasp, atoms, vasp5=True, sort=True)
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Wrote production trajectory to: {traj_path}")
    print(f"Wrote final structure to: {final_vasp}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
