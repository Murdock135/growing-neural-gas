import numpy as np
import matplotlib.pyplot as plt
import utils
from abc import ABC
from abc import abstractmethod
from typing import Any


class AdaptiveVectorQuantizer(ABC):
    def __init__(
            self,
            data: np.ndarray,
            results_dir: str,
            max_iter: int = 'auto',
            epochs: int = 3,
            plot_interval: int = 100,
            sampling_without_replacement: bool = True,
            **plotting_colors
    ) -> None:
        self.data: np.ndarray = data
        self.results_dir = results_dir
        self.max_iter = max_iter
        self.epochs = epochs
        self.plot_interval = plot_interval
        self.sampling_without_replacement = sampling_without_replacement
        self.plotting_colors: dict[str, str] = plotting_colors
        self.check_max_iter()
        self.fig, self.ax = plt.subplots()
    
    @abstractmethod
    def update(self):
        pass

    def get_data_size(self): return self.data.shape[0]

    def get_dim(self): return self.data.shape[1]

    def check_max_iter(self):
        if self.sampling_without_replacement == True and self.max_iter > self.data.shape[0]:
            self.sampling_without_replacement = False
            print("Max iter > Data size. Will sample with replacement")
    
    def shuffle_data(self):
        rng = np.random.default_rng()
        random_sequence = rng.choice(a=self.get_data_size(),
                                        size=self.max_iter,
                                        replace=not self.sampling_without_replacement)
        return self.data[random_sequence]
        