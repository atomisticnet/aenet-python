"""
Metrics tracking for PyTorch training.

Handles training history and metrics computation.
"""

from typing import Dict, List


class MetricsTracker:
    """
    Tracks training metrics and history.

    Maintains per-epoch metrics for energy RMSE, force RMSE, learning rate,
    and timing information.

    Parameters
    ----------
    track_detailed_timing : bool
        Whether to track detailed timing breakdowns (data loading, loss
        computation, optimizer steps).
    """

    def __init__(self, track_detailed_timing: bool = False):
        self.track_detailed_timing = track_detailed_timing
        self.history: Dict[str, List[float]] = {
            "train_energy_rmse": [],
            "test_energy_rmse": [],
            "train_force_rmse": [],
            "test_force_rmse": [],
            "learning_rates": [],
            "epoch_times": [],
            "epoch_forward_time": [],
            "epoch_backward_time": [],
        }

        if track_detailed_timing:
            self.history.update(
                {
                    "epoch_data_loading_time_train": [],
                    "epoch_loss_time_train": [],
                    "epoch_optimizer_time_train": [],
                    "epoch_data_loading_time_val": [],
                    "epoch_loss_time_val": [],
                    "epoch_optimizer_time_val": [],
                    "epoch_train_time": [],
                    "epoch_val_time": [],
                }
            )

    def update(
        self,
        train_energy_rmse: float,
        train_force_rmse: float,
        test_energy_rmse: float,
        test_force_rmse: float,
        learning_rate: float,
        epoch_time: float,
        forward_time: float,
        backward_time: float,
        train_timing: Dict[str, float],
        val_timing: Dict[str, float],
    ):
        """
        Record metrics for one epoch.

        Parameters
        ----------
        train_energy_rmse : float
            Training energy RMSE.
        train_force_rmse : float
            Training force RMSE (or NaN if not computed).
        test_energy_rmse : float
            Validation energy RMSE (or NaN if no validation).
        test_force_rmse : float
            Validation force RMSE (or NaN if not computed).
        learning_rate : float
            Current learning rate.
        epoch_time : float
            Total epoch time in seconds.
        forward_time : float
            Forward pass time in seconds.
        backward_time : float
            Backward pass time in seconds.
        train_timing : dict
            Detailed training timing breakdown with keys:
            'data_loading', 'loss_computation', 'optimizer', 'total'
        val_timing : dict
            Detailed validation timing breakdown (same keys as train_timing).
        """
        self.history["train_energy_rmse"].append(float(train_energy_rmse))
        self.history["train_force_rmse"].append(float(train_force_rmse))
        self.history["test_energy_rmse"].append(float(test_energy_rmse))
        self.history["test_force_rmse"].append(float(test_force_rmse))
        self.history["learning_rates"].append(float(learning_rate))
        self.history["epoch_times"].append(float(epoch_time))
        self.history["epoch_forward_time"].append(float(forward_time))
        self.history["epoch_backward_time"].append(float(backward_time))

        if self.track_detailed_timing:
            self.history["epoch_data_loading_time_train"].append(
                float(train_timing.get("data_loading", 0.0))
            )
            self.history["epoch_loss_time_train"].append(
                float(train_timing.get("loss_computation", 0.0))
            )
            self.history["epoch_optimizer_time_train"].append(
                float(train_timing.get("optimizer", 0.0))
            )
            self.history["epoch_train_time"].append(
                float(train_timing.get("total", 0.0))
            )

            self.history["epoch_data_loading_time_val"].append(
                float(val_timing.get("data_loading", 0.0))
            )
            self.history["epoch_loss_time_val"].append(
                float(val_timing.get("loss_computation", 0.0))
            )
            self.history["epoch_optimizer_time_val"].append(
                float(val_timing.get("optimizer", 0.0))
            )
            self.history["epoch_val_time"].append(
                float(val_timing.get("total", 0.0))
            )

    def get_history(self) -> Dict[str, List[float]]:
        """
        Get complete training history.

        Returns
        -------
        dict
            Dictionary mapping metric names to lists of per-epoch values.
        """
        return self.history

    def get_latest(self, metric: str) -> float:
        """
        Get the latest value for a specific metric.

        Parameters
        ----------
        metric : str
            Metric name (e.g., 'train_energy_rmse').

        Returns
        -------
        float
            Latest value, or NaN if metric not found or empty.
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return float("nan")
        return float(self.history[metric][-1])

    def get_best(self, metric: str, mode: str = "min") -> float:
        """
        Get the best value for a specific metric.

        Parameters
        ----------
        metric : str
            Metric name.
        mode : str
            Either 'min' or 'max'.

        Returns
        -------
        float
            Best value, or NaN if metric not found or empty.
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return float("nan")

        values = self.history[metric]
        if mode == "min":
            return float(min(values))
        elif mode == "max":
            return float(max(values))
        else:
            raise ValueError(f"Invalid mode '{mode}', use 'min' or 'max'")

    def reset(self):
        """Clear all metrics history."""
        for key in self.history:
            self.history[key] = []

    def set_history(self, history: Dict[str, List[float]]):
        """
        Set history from a dictionary (e.g., when loading checkpoint).

        Parameters
        ----------
        history : dict
            Dictionary mapping metric names to lists of values.
        """
        self.history = history
