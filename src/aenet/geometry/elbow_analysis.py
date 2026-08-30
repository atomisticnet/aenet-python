"""Plot a k-means elbow curve for representation matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def _validate_features(representations: np.ndarray, max_k: int) -> np.ndarray:
    features = np.asarray(representations, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("representations must be a non-empty 2D array.")
    if not np.all(np.isfinite(features)):
        raise ValueError("representations must contain only finite values.")
    if max_k < 1:
        raise ValueError("max_k must be positive.")
    if max_k > features.shape[0]:
        raise ValueError("max_k cannot exceed the number of samples.")
    return features


def _x_ticks(max_k: int, max_ticks: int = 10) -> np.ndarray:
    if max_k <= max_ticks:
        return np.arange(1, max_k + 1)
    return np.unique(np.round(np.linspace(1, max_k, max_ticks)).astype(int))


def elbow_inertias(
    representations: np.ndarray,
    max_k: int = 50,
    random_state: int | None = None,
    n_init: int | str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return k values and k-means inertias."""
    features = _validate_features(representations, max_k)
    k_values = np.arange(1, max_k + 1)
    inertias = np.array(
        [
            KMeans(n_clusters=int(k), random_state=random_state, n_init=n_init)
            .fit(features)
            .inertia_
            for k in k_values
        ]
    )
    return k_values, inertias


def plot_elbow_method(
    representations: np.ndarray,
    max_k: int = 50,
    random_state: int | None = None,
    n_init: int | str = "auto",
    output_path: str | Path = "elbow_analysis.png",
    show: bool = False,
) -> Path:
    """Save an elbow-curve PNG and return its path."""
    k_values, inertias = elbow_inertias(
        representations,
        max_k=max_k,
        random_state=random_state,
        n_init=n_init,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.plot(k_values, inertias, marker="o", markersize=3, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlim(1, max_k)
    ax.set_xticks(_x_ticks(max_k))
    ax.set_xlabel("Number of clusters, k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-means elbow analysis")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    """Run the elbow-analysis command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features", help="Path to .npy or .npz feature matrix."
    )
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--output", default="elbow_analysis.png")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.features)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            features = data["features"]
    else:
        features = data
    output_path = plot_elbow_method(
        features,
        max_k=args.max_k,
        random_state=args.random_state,
        output_path=args.output,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
