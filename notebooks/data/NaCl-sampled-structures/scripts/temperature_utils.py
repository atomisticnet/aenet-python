from __future__ import annotations

import re
from pathlib import Path


def temperature_name_from_parent(folder: Path | str) -> str:
    """Extract a numeric temperature from a directory's parent name."""
    parent_folder = Path(folder).parent.name
    match = re.fullmatch(r"(\d+)K", parent_folder, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Expected parent folder to be named like '700K', got: {parent_folder}"
        )
    return match.group(1)
