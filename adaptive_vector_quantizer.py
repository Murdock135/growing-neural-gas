import numpy as np
import utils.plot_utils as pu
from abc import ABC, abstractmethod
from typing import Any
from tqdm import tqdm


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
        **plotting_colors,
    ) -> None:
        self.data: np.ndarray = data
        self.neurons_n = neurons_n
        self.results_dir = results_dir
        self.lifetime = lifetime
        self.max_iter = max_iter
        self.epochs = epochs
        self.plot_interval = plot_interval
        self.sampling_without_replacement = sampling_without_replacement
        self.plotting_colors: dict[str, str] = plotting_colors
        self.check_max_iter()
        self.sample_counts = np.zeros(self.data.shape[0])
        # self.fig, self.ax = plt.subplots()

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
                    pu.plot_NG(
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
