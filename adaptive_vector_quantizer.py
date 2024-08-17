from pickle import TUPLE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
from sympy import true
from tqdm import tqdm
import utils.plot_utils as pu


class AdaptiveVectorQuantizer(ABC):
    def __init__(
        self,
        data: np.ndarray,
        neurons_n: int,
        results_dir: str,
        plotting_colors: Dict = pu.NG_colors,
        **kwargs,
    ) -> None:
        self.data: np.ndarray = data
        self.neurons_n = neurons_n
        self.results_dir = results_dir

        # kwargs
        self.lifetime = kwargs.get("lifetime", 10)
        self.max_iter = kwargs.get("max_iter", self.data.shape[0])
        self.epochs = kwargs.get("epochs", 3)
        self.plot_interval = kwargs.get("plot_interval", 100)
        self.sampling_without_replacement = kwargs.get(
            "sampling_without_replacement", True
        )
        self._correct_sampling_logic()

        # Additional initializations
        self.sample_counts = np.zeros(self.data.shape[0])
        self.color_dict = plotting_colors
        self.fig, self.ax = plt.subplots()
        self.color_dict = self.set_plotting_colors(**self.color_dict)
        self.colorbar = None

    def run(self):
        self.neurons: np.ndarray = self.create_neurons(self.neurons_n, dist="uniform")
        self.connection_matrix: np.ndarray = np.zeros((self.neurons_n, self.neurons_n))

        # create colorbar
        cmap = ListedColormap([self.color_dict["data"], "#ffd1df", "salmon", "red"])
        norm = BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=cmap.N)
        if self.colorbar is None:
            self.colorbar = self.create_cbar(cmap, norm)

        for epoch in tqdm(range(self.epochs), desc="Epoch"):
            # create random sequence
            random_seq = self.get_random_sequence(self.data)

            # create cmap
            if self.sampling_without_replacement == True:
                new_cmap_colors = [cmap(epoch), cmap(epoch + 1)]
                new_cmap = ListedColormap(new_cmap_colors)

            for i in tqdm(range(self.max_iter)):
                choice = random_seq[i]
                x = self.data[choice]
                self.sample_counts[choice] += 1
                self.update(i, x)

                if i % self.plot_interval == 0 or i == self.max_iter - 1:
                    self.ax.clear()
                    self.plot_NG(
                        data=self.data,
                        neurons=self.neurons,
                        sample_counts=self.sample_counts,
                        current_sample=x,
                        connection_matrix=self.connection_matrix,
                        epoch=epoch,
                        iter=i,
                        cmap=new_cmap,
                    )

                    pu.save_fig(self.results_dir, epoch, i)

            # Create GIF
            pu._create_gif(self.results_dir, epoch)

    @abstractmethod
    def update(self, i: int, x: np.ndarray):
        pass

    def _correct_sampling_logic(self):
        if (
            self.sampling_without_replacement == True
            and self.max_iter > self.data.shape[0]
        ):
            self.sampling_without_replacement = False
            print("Max iter > Data size. Will sample with replacement")
        if self.max_iter == "auto":
            assert (
                self.max_iter == self.data.shape[0]
            ), "Max iterations is set to 'auto'. Data size need to equal number of iterations (max_iter)"

    def get_random_sequence(self, data: np.ndarray):
        rng = np.random.default_rng()
        random_sequence = rng.choice(
            a=data.shape[0],
            size=self.max_iter,
            replace=not self.sampling_without_replacement,
        )
        return random_sequence

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

    def set_plotting_colors(self, **colors_kwargs) -> Dict:
        return {
            "data": colors_kwargs.get("data", "0.8"),
            "neurons": colors_kwargs.get("neurons", "k"),
            "current_sample_facecolor": colors_kwargs.get(
                "current_sample_facecolor", "green"
            ),
            "current_sample_edgecolor": colors_kwargs.get(
                "current_sample_edgecolor", "k"
            ),
            "connection": colors_kwargs.get("connection", "k"),
        }

    def plot_NG(
        self,
        data,
        neurons,
        sample_counts=None,
        current_sample=None,
        connection_matrix=None,
        epoch=None,
        iter=None,
        cmap=None,
    ):
        cur_sample_size = 30
        if sample_counts is not None:
            sc = self.ax.scatter(
                data[:, 0], data[:, 1], s=10, marker='o', c=sample_counts, cmap=cmap, label="Data"
            )
        else:
            sc = self.ax.scatter(data[:, 0], data[:, 1], label="Data")

        self.ax.scatter(
            neurons[:, 0], neurons[:, 1], s=10, marker='o', c=self.color_dict["neurons"], label="Neurons"
        )

        if current_sample is not None:
            self.ax.scatter(
                current_sample[0],
                current_sample[1],
                facecolor=self.color_dict["current_sample_facecolor"],
                edgecolor=self.color_dict["current_sample_edgecolor"],
                s=30,
                label="Current sample",
            )

        if connection_matrix is not None:
            n1s, n2s = np.nonzero(np.triu(connection_matrix, k=1))
            for n1, n2 in zip(n1s, n2s):
                neuron_1 = neurons[n1]
                neuron_2 = neurons[n2]
                x_coords = [neuron_1[0], neuron_2[0]]
                y_coords = [neuron_1[1], neuron_2[1]]
                self.ax.plot(
                    x_coords,
                    y_coords,
                    color=self.color_dict["connection"],
                    linestyle="-",
                )

        plt.title(f"Epoch {epoch}\nIteration {iter}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

    def create_cbar(self, cmap, norm):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        tick_pos = [0.5, 1.5, 2.5, 3.5]
        cbar = self.fig.colorbar(sm, ax=self.ax, ticks=tick_pos)
        cbar.set_ticklabels(list(range(self.epochs + 1)))
        cbar.set_label('Number of times sampled')
        return cbar