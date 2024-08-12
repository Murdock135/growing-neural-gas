# adaptive_vector_quantizer.py

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
import imageio
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import utils


class InvalidDataError(Exception):
    """Custom exception for invalid input data."""

    pass


class AdaptiveVectorQuantizer(ABC):
    """
    Abstract base class for Adaptive Vector Quantizer algorithms.
    """

    def __init__(
        self,
        data: np.ndarray,
        fig_save_path: str,
        max_iterations: int,
        max_epochs: int,
        plot_interval: int,
        sample_without_replacement: bool = True,
    ) -> None:
        """
        Initialize the Adaptive Vector Quantizer.

        Args:
            data (np.ndarray): Input data points.
            fig_save_path (str): Path to save output figures.
            neurons_n (int): Number of neurons.
            max_iterations (int or str): Maximum number of iterations per epoch.
            max_epochs (int): Number of epochs.
            plot_interval (int): Interval for plotting and saving figures.
            sample_without_replacement (bool): Whether to sample data without replacement.
        """

        # Check data type
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise InvalidDataError("Data must be a 2D numpy array.")

        # Algorithm metaparameters
        self.data: np.ndarray = data
        self.size: int = data.shape[0]
        self.dim: int = data.shape[1]
        self.fig_save_path: str = fig_save_path
        self.max_epochs: int = max_epochs
        self.max_iterations: int = (
            self.size if max_iterations == "auto" else max_iterations
        )
        self.plot_interval: int = plot_interval

        # Sample data
        if max_iterations != "auto" and max_iterations > self.size:
            self.sample_without_replacement = False
            if sample_without_replacement:
                print(
                    "Maximum number of iterations > Data size. Will sample with replacement"
                )
        else:
            self.sample_without_replacement = sample_without_replacement

        rng = np.random.default_rng(0)
        self.shuffled_indices = rng.choice(
            a=self.size,
            size=self.max_iterations,
            replace=not self.sample_without_replacement,
        )

        # Per-Iteration information
        self.sampled_indices: List[List[int]] = [[] for _ in range(self.max_epochs)]

        self.current_sample_idx: int = 0
        self.current_sample: np.ndarray = np.array([])
        self.current_epoch: int = 0
        self.current_iteration: int = 0

        # OS-level information
        self.debug: bool = os.environ.get("DEBUG", "False").lower() == "true"
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO")

        for i in range(1, self.max_epochs + 1):
            os.makedirs(os.path.join(self.fig_save_path, str(i)), exist_ok=True)

        # Plotting information
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.colorbar = None

    def run(self) -> None:
        """Run the adaptive vector quantization algorithm."""
        if self.debug:
            print(f"Running {self.__class__.__name__}")

        for e in tqdm(range(1, self.max_epochs + 1)):
            self.current_epoch = e
            self._run_local_iterations()
            self._create_gif()

        print("Run complete")

    def _run_local_iterations(self) -> None:
        """Run local iterations for the current epoch."""
        for i in tqdm(range(1, self.max_iterations + 1)):
            self.current_iteration = i
            k = i - 1
            self.current_sample_idx = self.shuffled_indices[k]
            self.current_sample = self.data[self.current_sample_idx]
            self.sampled_indices[self.current_epoch - 1].append(self.current_sample_idx)

            self._update()
            if i % self.plot_interval == 0 or i == self.max_iterations:
                self.save_plot(cmap=utils.cmap)

    @abstractmethod
    def _update(self) -> None:
        """Update method to be implemented by subclasses."""
        pass

    @abstractmethod
    def alter_topology(self, r_index: int, c_index: int) -> None:
        """
        Alter topology method to be implemented by subclasses.

        Args:
            r_index (int): Index of the first neuron.
            c_index (int): Index of the second neuron.
        """
        pass

    @abstractmethod
    def create_neurons(self, neurons_n: int, dist: str = "uniform") -> np.ndarray:
        """
        Create initial neuron positions.

        Args:
            neurons_n (int): Number of neurons to create.
            dist (str): Distribution to use for neuron positions ('uniform' or 'normal').

        Returns:
            np.ndarray: Array of neuron positions.
        """
        pass

    def plot(self, **kwargs: Any) -> plt.Figure:
        """
        Plot the current state of the algorithm.

        Keyword Args:
            neurons_color (str): Color for neuron markers. Default is 'k' (black).
            data_facecolor (str): Face color for data points. Default is '#4575F3' (blue).
            data_edgecolor (str): Edge color for data points. Default is 'none'.
            current_sample_color (str): Color for the current sample. Default is 'green'.
            connection_color (str): Color for neuron connections. Default is 'k' (black).
            cmap (str or Colormap): Colormap for sampled data points. Default is 'viridis'.
        Returns:
            plt.Figure: The plotted figure.
        """
        self.ax.clear()

        # Default colors
        default_colors: Dict[str, Any] = {
            "neurons_color": "k",
            "data_facecolor": "#4575F3",
            "data_edgecolor": "none",
            "current_sample_color": "green",
            "connection_color": "k",
            "cmap": "viridis",
        }

        # Use default colors if kwargs is empty, otherwise update with provided values
        colors = default_colors if not kwargs else {**default_colors, **kwargs}
        if isinstance(colors['cmap'], str):
            colors['cmap'] = plt.get_cmap(colors['cmap'])

        # Plot neurons
        self.ax.scatter(
            self.neuron_positions[:, 0],
            self.neuron_positions[:, 1],
            marker=".",
            label="neurons",
            c=colors["neurons_color"],
        )

        # Plot data
        self.ax.scatter(
            self.data[:, 0],
            self.data[:, 1],
            s=0.5,
            marker="o",
            label="data",
            facecolors=colors["data_facecolor"],
            edgecolors=colors["data_edgecolor"],
        )

        # Ensure length of cmap and # of lists of sampled points are compatible
        assert colors["cmap"].N >= len(self.sampled_indices)
        scatter_plots = []

        for i, list in enumerate(self.sampled_indices):
            if len(list) > 0:
                x_coords = self.data[list, 0]
                y_coords = self.data[list, 1]
                scatter = self.ax.scatter(
                    x_coords, y_coords, s=10, marker="o", facecolor=colors["cmap"](i)
                )
                scatter_plots.append(scatter)

        # Plot current sample
        self.ax.scatter(
            self.current_sample[0],
            self.current_sample[1],
            s=50,
            marker="o",
            label="current sample",
            facecolor=colors["current_sample_color"],
            edgecolor="k",
        )

        # Plot connections between neurons
        for r in range(self.connection_matrix.shape[0]):
            for c in range(r + 1, self.connection_matrix.shape[1]):
                if self.connection_matrix[r, c] > 0:
                    neuron_a = self.neuron_positions[r]
                    neuron_b = self.neuron_positions[c]
                    self.ax.plot(
                        [neuron_a[0], neuron_b[0]],
                        [neuron_a[1], neuron_b[1]],
                        color=colors["connection_color"],
                    )

        self.ax.legend()
        self.ax.set_title(
            f"Epoch: {self.current_epoch}\nIteration: {self.current_iteration}"
        )

        # Create or update colorbar
        cmap_obj = colors['cmap']
        cmap_obj = cmap_obj.with_extremes(under=colors["data_facecolor"], over="red")
        bounds = np.arange(self.max_epochs + 1).tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap_obj.N)
        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(sm, ax=self.ax, extend="both", spacing="proportional")
        else:
            self.colorbar.update_normal(sm)

        self.colorbar.set_label("Number of times sampled")
        return self.fig

    def save_plot(self, **kwargs) -> None:
        """Save the current plot to a file."""
        self.plot(**kwargs)
        file_name = os.path.join(
            self.fig_save_path,
            str(self.current_epoch),
            f"frame_{self.current_iteration:04d}.png",
        )
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.fig.savefig(file_name)

    def _create_gif(self) -> None:
        """Create a GIF from the saved plots for the current epoch."""
        figures = []
        results_for_epoch = os.path.join(self.fig_save_path, str(self.current_epoch))
        filenames = sorted(os.listdir(results_for_epoch))

        for f in filenames:
            filepath = os.path.join(self.fig_save_path, str(self.current_epoch), f)
            figures.append(imageio.imread(filepath))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(
            self.fig_save_path,
            str(self.current_epoch),
            f"{self.__class__.__name__}_animation_{timestamp}.gif",
        )
        imageio.mimsave(gif_path, figures, duration=0.1)
