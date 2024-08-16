from ast import Dict, Tuple
import matplotlib
import matplotlib.axes
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
import numpy as np
import utils.plot_utils as pu
from abc import ABC, abstractmethod
from typing import Dict
from tqdm import tqdm
from utils.plot_utils import sample_count_colors, NG_colors
import matplotlib.pyplot as plt


class AdaptiveVectorQuantizer(ABC):
    def __init__(
        self,
        data: np.ndarray,
        neurons_n: int,
        results_dir: str,
        lifetime: int = "auto",
        max_iter: int = "auto",
        epochs: int = 3,
        plot_interval: int = 100,
        sampling_without_replacement: bool = True,
        plot = True,
        plotting_colors: dict = None
    ) -> None:
        self.data: np.ndarray = data
        self.neurons_n = neurons_n
        self.results_dir = results_dir
        self.lifetime = lifetime
        self.max_iter = max_iter
        self.epochs = epochs
        self.plot_interval = plot_interval
        self.sampling_without_replacement = sampling_without_replacement
        self.check_max_iter()
        self.sample_counts = np.zeros(self.data.shape[0])
        if plot:
            self.color_dict = plotting_colors
            self.fig, self.ax = plt.subplots()
            self.colorbar = None
            self.cmap, self.color_dict = self.set_plotting_colors(self.color_dict)


class AdaptiveVectorQuantizer(ABC):
    def __init__(
        self,
        data: np.ndarray,
        neurons_n: int,
        results_dir: str,
        lifetime: int = "auto",
        max_iter: int = "auto",
        epochs: int = 3,
        plot_interval: int = 100,
        sampling_without_replacement: bool = True,
        plotting_colors: dict = None
    ) -> None:
        self.data: np.ndarray = data
        self.neurons_n = neurons_n
        self.results_dir = results_dir
        self.lifetime = lifetime
        self.max_iter = max_iter
        self.epochs = epochs
        self.plot_interval = plot_interval
        self.sampling_without_replacement = sampling_without_replacement
        self.check_max_iter()
        self.sample_counts = np.zeros(self.data.shape[0])
        self.color_dict = plotting_colors if plotting_colors else NG_colors
        print(self.color_dict)
        self.fig, self.ax = plt.subplots()
        self.cmap, self.color_dict = self.set_plotting_colors(self.color_dict)
        cbar_bounds = np.arange(self.epochs + 2)
        self.colorbar = self.set_colorbar(self.ax, self.cmap, cbar_bounds)

    def run(self):
        shuffled_data: np.ndarray = self.shuffle_data(self.data)
        self.neurons: np.ndarray = self.create_neurons(self.neurons_n, dist='uniform')
        self.connection_matrix: np.ndarray = np.zeros((self.neurons_n, self.neurons_n))
        if self.max_iter == "auto":
            assert (
                self.max_iter == self.data.shape[0]
            ), "Max iterations is set to 'auto'. Data size need to equal number of iterations (max_iter)"

        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            for i in tqdm(range(self.max_iter)):
                x = shuffled_data[i]
                data_idx = np.nonzero(self.data == x)[0]
                self.sample_counts[data_idx] += 1
                self.update(i, x)

                if i % self.plot_interval == 0 or i == self.max_iter - 1:
                    self.plot_NG(
                        data=self.data,
                        neurons=self.neurons,
                        current_sample=x,
                        sample_counts=self.sample_counts,
                        iter=i,
                        epoch=epoch,
                        connection_matrix=self.connection_matrix
                    )
                    pu.save_fig(self.results_dir, epoch, i)

    @abstractmethod
    def update(self, i: int, x: np.ndarray):
        pass

    def check_max_iter(self):
        if (
            self.sampling_without_replacement == True
            and self.max_iter > self.data.shape[0]
        ):
            self.sampling_without_replacement = False
            print("Max iter > Data size. Will sample with replacement")

    def shuffle_data(self, data: np.ndarray):
        rng = np.random.default_rng()
        random_sequence = rng.choice(
            a=data.shape[0],
            size=self.max_iter,
            replace=not self.sampling_without_replacement,
        )
        return data[random_sequence]

    def create_neurons(self, neurons_n, dist: str) -> np.ndarray:
        dim = self.data.shape[1]
        min_values = np.amin(self.data, axis=0)
        max_values = np.amax(self.data, axis=0)
        rng = np.random.default_rng(0)
        if dist.lower() == "uniform":
            return rng.uniform(low=min_values, high=max_values, size=(neurons_n, dim))
        elif dist.lower() == "normal":
            mean_arr = np.mean(self.data, axis=0)
            std_arr = np.std(self.data, axis=0)
            return rng.normal(mean_arr, std_arr, size=(neurons_n, dim))
        else:
            print(
                f"Distribution {dist} not recognized. Use either 'uniform' or 'normal'"
            )

    def increase_age(self, r_index, c_index):
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            self.connection_matrix[r_index, c_index] += 1
            self.connection_matrix[c_index, r_index] += 1

    def remove_old_connections(self):
        """Remove connections older than the specified lifetime and delete lonely neurons."""
        old_connections = self.connection_matrix > self.lifetime
        self.connection_matrix[old_connections] = 0
    
    def set_plotting_colors(self, color_dict=None) -> tuple[ListedColormap, Dict, ]:
        colors = color_dict if color_dict is not None else NG_colors
        cmap_colors_copy = colors['sample_count_colors'].copy()
        cmap_colors_copy.insert(0, colors['data'])
        cmap = ListedColormap(cmap_colors_copy)

        assert (
            cmap.N == self.epochs + 1
        ), f"Number of colors in the color map ({cmap.N}) must be one more than number of epochs ({self.epochs})"

        return (cmap, color_dict)
    
    def set_colorbar(self, ax: matplotlib.axes, cmap: ListedColormap, bounds: list):
        norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
        sm =  plt.cm.ScalarMappable(cmap = cmap, norm=norm)
        sm.set_array([])
        print(bounds)
        cbar = plt.colorbar(sm, ax=ax, ticks=bounds)
        cbar.set_ticks(bounds[:-1] + 0.5)  # Center the ticks between the boundaries
        cbar.set_ticklabels(bounds[:-1])   # Label each segment

        return cbar
        
    def plot_NG(
        self,
        data,
        neurons,
        iter,
        epoch,
        current_sample= None,
        sample_counts: np.ndarray = None,
        connection_matrix: np.ndarray = None,
    ):
        self.ax.clear()
        colors = self.color_dict
        cmap = self.cmap
        size = 50
        sc = plt.scatter(
            data[:, 0],
            data[:, 1],
            c=sample_counts,
            cmap=cmap,
            marker="o",
            s=size / 3,
            label="Data",
        )
        plt.scatter(
            neurons[:, 0], neurons[:, 1], c=colors["neurons"], marker=".", label="Neurons"
        )
        if current_sample is not None:
            plt.scatter(
                current_sample[0],
                current_sample[1],
                facecolor=colors["current_sample_facecolor"],
                edgecolors=colors["current_sample_edgecolor"],
                marker="o",
                s=size,
                label="Current sample",
            )

        if connection_matrix is not None:
            n1s, n2s = np.nonzero(np.triu(connection_matrix, k=1))
            for n1, n2 in zip(n1s, n2s):
                neuron_1 = neurons[n1]
                neuron_2 = neurons[n2]
                x_coords = [neuron_1[0], neuron_2[0]]
                y_coords = [neuron_1[1], neuron_2[1]]
                plt.plot(x_coords, y_coords, color=colors["connection"], linestyle="-")

        plt.title(f"Epoch {epoch}\nIteration {iter}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        plt.draw()