#!/usr/bin/env python3
"""Convert one UMA production trajectory to temperature-labeled XSF files."""

import argparse
from pathlib import Path

from temperature_utils import temperature_name_from_parent


def snapshot_name(output_index: int, temperature: str) -> str:
    """Return the filename used by the tracked candidate dataset."""
    return f"snapshot_{output_index:04d}_{temperature}K.xsf"


def main() -> None:
    """Convert all production frames after frame zero to XSF files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("sampled_structures")
    )
    args = parser.parse_args()

    from ase.io.trajectory import Trajectory

    import aenet.geometry
    import aenet.io.structure

    temperature = temperature_name_from_parent(Path.cwd())
    trajectory_path = args.trajectory or Path(
        f"production_{temperature}K.traj"
    )
    trajectory = Trajectory(str(trajectory_path))
    if len(trajectory) < 2:
        raise ValueError(
            "trajectory must contain equilibration frame 0 and production frames"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    production_frames = range(1, len(trajectory))
    print(
        f"Converting {len(trajectory) - 1} production frames from {trajectory_path}"
    )

    for output_index, frame_index in enumerate(production_frames, start=1):
        structure = aenet.geometry.AtomicStructure.from_ase_atoms(
            trajectory[frame_index]
        )
        output_path = args.output_dir / snapshot_name(
            output_index, temperature
        )
        aenet.io.structure.write(structure, str(output_path))

        if output_index % 500 == 0:
            print(f"Written {output_index}/{len(trajectory) - 1} snapshots")

    print(f"Wrote {len(trajectory) - 1} XSF files to {args.output_dir}")


if __name__ == "__main__":
    main()
