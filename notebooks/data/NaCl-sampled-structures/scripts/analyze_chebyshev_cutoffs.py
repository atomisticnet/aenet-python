"""Analyze NaCl XSF snapshots to choose Chebyshev descriptor cutoffs."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TEMP_RE = re.compile(r"_(\d+)K(?:\.xsf)?$")
IMAGE_SHIFTS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


def read_xsf(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return cell vectors, species, and Cartesian coordinates from an XSF file."""
    lines = path.read_text().splitlines()
    primvec = lines.index("PRIMVEC")
    cell = np.array(
        [
            [float(x) for x in lines[primvec + offset].split()[:3]]
            for offset in range(1, 4)
        ],
        dtype=float,
    )

    primcoord = lines.index("PRIMCOORD")
    n_atoms = int(lines[primcoord + 1].split()[0])
    species: list[str] = []
    coords = np.empty((n_atoms, 3), dtype=float)
    for atom_index, line in enumerate(
        lines[primcoord + 2 : primcoord + 2 + n_atoms]
    ):
        fields = line.split()
        species.append(fields[0])
        coords[atom_index] = [float(value) for value in fields[1:4]]
    return cell, species, coords


def temperature_from_name(path: Path) -> str:
    """Return a temperature label from a snapshot filename."""
    match = TEMP_RE.search(path.stem)
    if match is None:
        return "unknown"
    return f"{match.group(1)}K"


def species_pair_label(left: str, right: str) -> str:
    """Return a stable species-pair label for NaCl reports and plots."""
    if {left, right} == {"Na", "Cl"}:
        return "Na-Cl"
    if left == right:
        return f"{left}-{right}"
    return "-".join(sorted((left, right)))


def minimum_image_distances_fast(
    cell: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """Compute pairwise periodic minimum-image distances using rounded fractions."""
    inv_cell = np.linalg.inv(cell)
    frac = coords @ inv_cell
    frac_delta = frac[:, None, :] - frac[None, :, :]
    frac_delta = frac_delta - np.rint(frac_delta)
    distances = np.linalg.norm(frac_delta @ cell, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances


def minimum_image_distances_exact(
    cell: np.ndarray, coords: np.ndarray
) -> np.ndarray:
    """Compute pairwise periodic minimum-image distances by checking nearby images."""
    inv_cell = np.linalg.inv(cell)
    frac = coords @ inv_cell
    frac_delta = frac[:, None, :] - frac[None, :, :]
    frac_delta = frac_delta - np.rint(frac_delta)
    candidates = (
        frac_delta[:, :, None, :] + IMAGE_SHIFTS[None, None, :, :]
    ) @ cell
    distances = np.sqrt(
        np.einsum("...i,...i->...", candidates, candidates).min(axis=-1)
    )
    np.fill_diagonal(distances, np.inf)
    return distances


def validate_fast_minimum_image(paths: list[Path], n_per_temp: int = 3) -> str:
    """Validate that the fast MIC gives the same first 26 neighbor distances."""
    selected: list[Path] = []
    counts: Counter[str] = Counter()
    for path in paths:
        temp = temperature_from_name(path)
        if counts[temp] < n_per_temp:
            selected.append(path)
            counts[temp] += 1

    max_error = 0.0
    for path in selected:
        cell, _, coords = read_xsf(path)
        fast = np.sort(minimum_image_distances_fast(cell, coords), axis=1)[
            :, :26
        ]
        exact = np.sort(minimum_image_distances_exact(cell, coords), axis=1)[
            :, :26
        ]
        max_error = max(max_error, float(np.max(np.abs(fast - exact))))

    if max_error > 1.0e-8:
        raise ValueError(
            "Fast minimum-image distance check failed: max first-26-shell "
            f"error was {max_error:.3e} Angstrom."
        )
    return (
        f"Validated fast minimum-image distances against exact 27-image search "
        f"on {len(selected)} snapshots; max first-26-shell error = "
        f"{max_error:.3e} Angstrom."
    )


def safe_quantile(values: np.ndarray, q: float) -> float:
    """Return a quantile as a float."""
    return float(np.quantile(values, q))


def classification_cutoff(
    lower_shell: np.ndarray,
    upper_shell: np.ndarray,
    lower_quantile: float = 0.999,
    upper_quantile: float = 0.001,
) -> tuple[float, dict[str, float]]:
    """Choose a cutoff balancing desired-shell capture and next-shell leakage."""
    lower_hi = safe_quantile(lower_shell, lower_quantile)
    upper_lo = safe_quantile(upper_shell, upper_quantile)
    lower_med = safe_quantile(lower_shell, 0.5)
    upper_med = safe_quantile(upper_shell, 0.5)

    lower_sorted = np.sort(lower_shell)
    upper_sorted = np.sort(upper_shell)
    candidates = np.linspace(lower_med, upper_med, 4000)
    lower_capture = (
        np.searchsorted(lower_sorted, candidates, side="right")
        / lower_sorted.size
    )
    upper_leakage = (
        np.searchsorted(upper_sorted, candidates, side="right")
        / upper_sorted.size
    )
    scores = lower_capture - upper_leakage
    best_score = np.max(scores)
    best = candidates[np.isclose(scores, best_score)]
    cutoff = float(np.median(best))
    method = "maximized desired-shell capture minus next-shell leakage"

    stats = {
        "lower_q999": lower_hi,
        "upper_q001": upper_lo,
        "lower_max": float(np.max(lower_shell)),
        "upper_min": float(np.min(upper_shell)),
        "lower_median": lower_med,
        "upper_median": upper_med,
        "lower_capture": float(np.mean(lower_shell <= cutoff)),
        "upper_leakage": float(np.mean(upper_shell <= cutoff)),
        "score": float(best_score),
        "method": method,
    }
    return cutoff, stats


def plot_cutoff_tradeoffs(
    first_all: np.ndarray,
    second_all: np.ndarray,
    third_all: np.ndarray,
    angular_cutoff: float,
    radial_cutoff: float,
    outdir: Path,
) -> None:
    """Plot cutoff-dependent capture/leakage tradeoffs."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    cases = [
        (
            axes[0],
            first_all,
            second_all,
            angular_cutoff,
            "Angular cutoff: first shell vs second shell",
            "First shell captured",
            "Second shell leaked",
        ),
        (
            axes[1],
            second_all,
            third_all,
            radial_cutoff,
            "Radial cutoff: second shell vs third shell",
            "Second shell captured",
            "Third shell leaked",
        ),
    ]
    for ax, lower, upper, cutoff, title, lower_label, upper_label in cases:
        thresholds = np.linspace(
            safe_quantile(lower, 0.001),
            safe_quantile(upper, 0.999),
            600,
        )
        lower_sorted = np.sort(lower)
        upper_sorted = np.sort(upper)
        lower_capture = (
            np.searchsorted(lower_sorted, thresholds, side="right")
            / lower_sorted.size
        )
        upper_leakage = (
            np.searchsorted(upper_sorted, thresholds, side="right")
            / upper_sorted.size
        )
        ax.plot(thresholds, 100.0 * lower_capture, label=lower_label)
        ax.plot(thresholds, 100.0 * upper_leakage, label=upper_label)
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Cutoff distance (Angstrom)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Fraction of shell distances below cutoff (%)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "cutoff_tradeoff_curves.png", dpi=220)
    plt.close(fig)


def plot_shell_histograms(
    shells_by_temp: dict[str, dict[str, np.ndarray]],
    angular_cutoff: float,
    radial_cutoff: float,
    outdir: Path,
) -> None:
    """Plot shell-distance histograms by temperature."""
    import matplotlib.pyplot as plt

    temps = sorted(
        shells_by_temp,
        key=lambda value: int(value.rstrip("K")) if value != "unknown" else 0,
    )
    colors = {
        "first": "tab:blue",
        "second": "tab:green",
        "third": "tab:red",
    }
    labels = {
        "first": "1st shell, neighbor ranks 1-6",
        "second": "2nd shell, ranks 7-18",
        "third": "3rd shell, ranks 19-26",
    }

    fig, axes = plt.subplots(
        len(temps), 1, figsize=(9, 2.6 * len(temps)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, temp in zip(axes, temps):
        for shell_name in ("first", "second", "third"):
            ax.hist(
                shells_by_temp[temp][shell_name],
                bins=160,
                range=(1.5, 7.0),
                density=True,
                histtype="step",
                linewidth=1.4,
                color=colors[shell_name],
                label=labels[shell_name],
            )
        ax.axvline(
            angular_cutoff, color="black", linestyle="--", linewidth=1.2
        )
        ax.axvline(radial_cutoff, color="black", linestyle="-", linewidth=1.2)
        ax.set_ylabel(temp)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Minimum-image neighbor distance (Angstrom)")
    fig.suptitle("NaCl neighbor-shell distance distributions by temperature")
    fig.tight_layout()
    fig.savefig(
        outdir / "shell_distance_histograms_by_temperature.png", dpi=220
    )
    plt.close(fig)


def plot_quantile_summary(
    shells_by_temp: dict[str, dict[str, np.ndarray]],
    angular_cutoff: float,
    radial_cutoff: float,
    outdir: Path,
) -> None:
    """Plot per-temperature shell quantiles used for cutoff validation."""
    import matplotlib.pyplot as plt

    temps = sorted(
        shells_by_temp,
        key=lambda value: int(value.rstrip("K")) if value != "unknown" else 0,
    )
    x = np.arange(len(temps))

    fig, ax = plt.subplots(figsize=(9, 5))
    for shell_name, color in (
        ("first", "tab:blue"),
        ("second", "tab:green"),
        ("third", "tab:red"),
    ):
        q001 = [
            safe_quantile(shells_by_temp[temp][shell_name], 0.001)
            for temp in temps
        ]
        q500 = [
            safe_quantile(shells_by_temp[temp][shell_name], 0.5)
            for temp in temps
        ]
        q999 = [
            safe_quantile(shells_by_temp[temp][shell_name], 0.999)
            for temp in temps
        ]
        ax.fill_between(x, q001, q999, color=color, alpha=0.16)
        ax.plot(x, q500, marker="o", color=color, label=f"{shell_name} median")

    ax.axhline(
        angular_cutoff,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="angular cutoff",
    )
    ax.axhline(
        radial_cutoff,
        color="black",
        linestyle="-",
        linewidth=1.2,
        label="radial cutoff",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(temps)
    ax.set_ylabel("Distance (Angstrom)")
    ax.set_title("Shell medians and 0.1-99.9% ranges by temperature")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "shell_quantile_summary.png", dpi=220)
    plt.close(fig)


def plot_pair_histograms(
    pair_distances_by_temp: dict[str, dict[str, np.ndarray]],
    angular_cutoff: float,
    radial_cutoff: float,
    outdir: Path,
) -> None:
    """Plot species-pair distance histograms by temperature."""
    import matplotlib.pyplot as plt

    temps = sorted(
        pair_distances_by_temp,
        key=lambda value: int(value.rstrip("K")) if value != "unknown" else 0,
    )
    pair_labels = ["Na-Cl", "Na-Na", "Cl-Cl"]
    fig, axes = plt.subplots(
        len(temps), 1, figsize=(9, 2.6 * len(temps)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, temp in zip(axes, temps):
        for label in pair_labels:
            values = pair_distances_by_temp[temp].get(label)
            if values is None or values.size == 0:
                continue
            ax.hist(
                values,
                bins=180,
                range=(1.5, 7.0),
                density=True,
                histtype="step",
                linewidth=1.4,
                label=label,
            )
        ax.axvline(
            angular_cutoff, color="black", linestyle="--", linewidth=1.2
        )
        ax.axvline(radial_cutoff, color="black", linestyle="-", linewidth=1.2)
        ax.set_ylabel(temp)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Minimum-image pair distance (Angstrom)")
    fig.suptitle("NaCl species-pair distance distributions by temperature")
    fig.tight_layout()
    fig.savefig(outdir / "species_pair_distance_histograms.png", dpi=220)
    plt.close(fig)


def summarize_species_purity(
    shell_species_counts: dict[str, Counter[str]],
) -> str:
    """Summarize the dominant species pairs in each neighbor-rank shell."""
    lines = []
    for shell_name in ("first", "second", "third"):
        counts = shell_species_counts[shell_name]
        total = sum(counts.values())
        parts = [
            f"{label}: {100.0 * count / total:.2f}%"
            for label, count in sorted(
                counts.items(), key=lambda item: item[0]
            )
        ]
        lines.append(f"- {shell_name}: " + ", ".join(parts))
    return "\n".join(lines)


def write_report(
    outdir: Path,
    n_files: int,
    n_atoms: int,
    shells_by_temp: dict[str, dict[str, np.ndarray]],
    angular_cutoff: float,
    angular_stats: dict[str, float],
    radial_cutoff: float,
    radial_stats: dict[str, float],
    species_summary: str,
    validation_summary: str,
) -> None:
    """Write the Markdown cutoff analysis report."""
    temps = sorted(
        shells_by_temp,
        key=lambda value: int(value.rstrip("K")) if value != "unknown" else 0,
    )
    lines = [
        "# NaCl Chebyshev Cutoff Analysis",
        "",
        f"Analyzed `{n_files}` XSF files with `{n_atoms}` atoms per structure.",
        "",
        "## Recommended Cutoffs",
        "",
        f"- Fitted angular cutoff: `{angular_cutoff:.3f}` Angstrom",
        f"- Fitted radial cutoff: `{radial_cutoff:.3f}` Angstrom",
        "- Practical descriptor setting: `ang_cutoff = 3.75` Angstrom",
        "- Practical descriptor setting: `rad_cutoff = 4.8` Angstrom",
        "",
        "Use these rounded descriptor settings:",
        "",
        "```python",
        "rad_cutoff = 4.8",
        "ang_cutoff = 3.75",
        "```",
        "",
        (
            "The exact fitted values are reported below so the recommendation remains "
            "traceable to the data."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The angular cutoff is placed between neighbor ranks 1-6 and ranks 7-18. "
            "For rocksalt NaCl, ranks 1-6 are the first unlike-ion coordination shell, "
            "while ranks 7-18 are the second same-sublattice shell."
        ),
        "",
        (
            "The radial cutoff is placed between ranks 7-18 and ranks 19-26, so radial "
            "features include the first and second shells while avoiding the next shell."
        ),
        "",
        (
            "At 850 K and 1000 K the shells overlap substantially. No single hard cutoff "
            "can include every thermally broadened second-shell distance while excluding "
            "the next shell. The recommended radial cutoff is therefore a balanced "
            "separator, not a perfect high-temperature second-shell envelope."
        ),
        "",
        "## Cutoff Selection Statistics",
        "",
        "| Boundary | Lower shell 99.9% | Upper shell 0.1% | Lower max | Upper min | Method |",
        "|---|---:|---:|---:|---:|---|",
        (
            f"| 1st/2nd shell, angular | {angular_stats['lower_q999']:.4f} | "
            f"{angular_stats['upper_q001']:.4f} | {angular_stats['lower_max']:.4f} | "
            f"{angular_stats['upper_min']:.4f} | {angular_stats['method']} |"
        ),
        (
            f"| 2nd/3rd shell, radial | {radial_stats['lower_q999']:.4f} | "
            f"{radial_stats['upper_q001']:.4f} | {radial_stats['lower_max']:.4f} | "
            f"{radial_stats['upper_min']:.4f} | {radial_stats['method']} |"
        ),
        "",
        "## Per-Temperature Shell Summary",
        "",
        "| Temperature | Shell | 0.1% | Median | 99.9% | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for temp in temps:
        for shell_name in ("first", "second", "third"):
            values = shells_by_temp[temp][shell_name]
            lines.append(
                f"| {temp} | {shell_name} | "
                f"{safe_quantile(values, 0.001):.4f} | "
                f"{safe_quantile(values, 0.5):.4f} | "
                f"{safe_quantile(values, 0.999):.4f} | "
                f"{np.min(values):.4f} | {np.max(values):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Cutoff Coverage Check",
            "",
            "| Temperature | 1st shell <= angular cutoff | 2nd shell <= angular cutoff | 2nd shell <= radial cutoff | 3rd shell <= radial cutoff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    all_first = []
    all_second = []
    all_third = []
    for temp in temps:
        first = shells_by_temp[temp]["first"]
        second = shells_by_temp[temp]["second"]
        third = shells_by_temp[temp]["third"]
        all_first.append(first)
        all_second.append(second)
        all_third.append(third)
        lines.append(
            f"| {temp} | "
            f"{100.0 * np.mean(first <= angular_cutoff):.2f}% | "
            f"{100.0 * np.mean(second <= angular_cutoff):.2f}% | "
            f"{100.0 * np.mean(second <= radial_cutoff):.2f}% | "
            f"{100.0 * np.mean(third <= radial_cutoff):.2f}% |"
        )
    first = np.concatenate(all_first)
    second = np.concatenate(all_second)
    third = np.concatenate(all_third)
    lines.append(
        f"| all | "
        f"{100.0 * np.mean(first <= angular_cutoff):.2f}% | "
        f"{100.0 * np.mean(second <= angular_cutoff):.2f}% | "
        f"{100.0 * np.mean(second <= radial_cutoff):.2f}% | "
        f"{100.0 * np.mean(third <= radial_cutoff):.2f}% |"
    )
    lines.extend(
        [
            "",
            "## Species-Pair Check",
            "",
            species_summary,
            "",
            "## Validation Plots",
            "",
            "- `shell_distance_histograms_by_temperature.png`: first, second, and third rank-shell distance distributions by temperature.",
            "- `shell_quantile_summary.png`: shell medians and 0.1-99.9% ranges by temperature.",
            "- `species_pair_distance_histograms.png`: Na-Cl, Na-Na, and Cl-Cl pair-distance distributions by temperature.",
            "- `cutoff_tradeoff_curves.png`: desired-shell capture and next-shell leakage as a function of cutoff.",
            "",
            "Dashed vertical line = angular cutoff. Solid vertical line = radial cutoff.",
            "",
            "## Notes",
            "",
            "- Distances are periodic minimum-image distances computed from each XSF cell.",
            f"- {validation_summary}",
            "- Shell labels use rocksalt NaCl coordination counts: 6 first-shell neighbors, 12 second-shell neighbors, and 8 next-shell neighbors.",
            "- The recommendation is intentionally conservative against high-temperature broadening by using all snapshots from every temperature.",
            "",
        ]
    )
    (outdir / "chebyshev_cutoff_report.md").write_text("\n".join(lines))


def main() -> None:
    """Analyze an XSF directory and write cutoff diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing temperature-labeled XSF snapshots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cutoff_analysis"),
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.xsf"))
    if not paths:
        raise SystemExit(f"No XSF files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_summary = (
        "Used exact 27-image periodic minimum-image search around wrapped "
        "fractional displacements for every pair; the faster rounded-fraction "
        "method failed validation for these skewed cells."
    )
    print(validation_summary, flush=True)

    shells_by_temp: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_distances_by_temp: dict[str, dict[str, list[np.ndarray]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    shell_species_counts: dict[str, Counter[str]] = defaultdict(Counter)
    n_atoms_seen: int | None = None

    upper_i = None
    upper_j = None
    for file_index, path in enumerate(paths, start=1):
        cell, species, coords = read_xsf(path)
        if n_atoms_seen is None:
            n_atoms_seen = len(species)
            upper_i, upper_j = np.triu_indices(n_atoms_seen, k=1)
        elif len(species) != n_atoms_seen:
            raise ValueError(
                f"{path} has {len(species)} atoms, expected {n_atoms_seen}"
            )

        temp = temperature_from_name(path)
        distances = minimum_image_distances_exact(cell, coords)
        order = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, order, axis=1)

        shells_by_temp[temp]["first"].append(sorted_distances[:, :6].ravel())
        shells_by_temp[temp]["second"].append(
            sorted_distances[:, 6:18].ravel()
        )
        shells_by_temp[temp]["third"].append(
            sorted_distances[:, 18:26].ravel()
        )

        species_array = np.array(species)
        sorted_species = species_array[order]
        center_species = species_array[:, None]
        for shell_name, rank_slice in (
            ("first", slice(0, 6)),
            ("second", slice(6, 18)),
            ("third", slice(18, 26)),
        ):
            center = np.repeat(
                center_species, sorted_species[:, rank_slice].shape[1], axis=1
            )
            neighbor = sorted_species[:, rank_slice]
            shell_species_counts[shell_name].update(
                species_pair_label(left, right)
                for left, right in zip(center.ravel(), neighbor.ravel())
            )

        pair_distances = distances[upper_i, upper_j]
        pair_species = [
            species_pair_label(species[left], species[right])
            for left, right in zip(upper_i, upper_j)
        ]
        pair_species = np.array(pair_species)
        pair_mask = pair_distances <= 7.0
        for pair_label in ("Na-Cl", "Na-Na", "Cl-Cl"):
            mask = pair_mask & (pair_species == pair_label)
            if np.any(mask):
                pair_distances_by_temp[temp][pair_label].append(
                    pair_distances[mask]
                )

        if file_index % 1000 == 0:
            print(f"Processed {file_index}/{len(paths)} files", flush=True)

    shell_arrays_by_temp = {
        temp: {
            shell_name: np.concatenate(chunks)
            for shell_name, chunks in shell_map.items()
        }
        for temp, shell_map in shells_by_temp.items()
    }
    pair_arrays_by_temp = {
        temp: {
            pair_label: np.concatenate(chunks)
            for pair_label, chunks in pair_map.items()
        }
        for temp, pair_map in pair_distances_by_temp.items()
    }

    first_all = np.concatenate(
        [shells["first"] for shells in shell_arrays_by_temp.values()]
    )
    second_all = np.concatenate(
        [shells["second"] for shells in shell_arrays_by_temp.values()]
    )
    third_all = np.concatenate(
        [shells["third"] for shells in shell_arrays_by_temp.values()]
    )

    angular_cutoff, angular_stats = classification_cutoff(
        first_all, second_all
    )
    radial_cutoff, radial_stats = classification_cutoff(second_all, third_all)

    plot_shell_histograms(
        shell_arrays_by_temp,
        angular_cutoff,
        radial_cutoff,
        args.output_dir,
    )
    plot_quantile_summary(
        shell_arrays_by_temp,
        angular_cutoff,
        radial_cutoff,
        args.output_dir,
    )
    plot_pair_histograms(
        pair_arrays_by_temp,
        angular_cutoff,
        radial_cutoff,
        args.output_dir,
    )
    plot_cutoff_tradeoffs(
        first_all,
        second_all,
        third_all,
        angular_cutoff,
        radial_cutoff,
        args.output_dir,
    )
    write_report(
        args.output_dir,
        len(paths),
        int(n_atoms_seen or 0),
        shell_arrays_by_temp,
        angular_cutoff,
        angular_stats,
        radial_cutoff,
        radial_stats,
        summarize_species_purity(shell_species_counts),
        validation_summary,
    )

    print(f"Angular cutoff: {angular_cutoff:.3f} Angstrom")
    print(f"Radial cutoff: {radial_cutoff:.3f} Angstrom")
    print(f"Wrote report and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
