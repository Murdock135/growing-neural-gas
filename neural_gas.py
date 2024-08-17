import numpy as np
from adaptive_vector_quantizer import AdaptiveVectorQuantizer
from utils.plot_utils import NG_colors


class NeuralGas(AdaptiveVectorQuantizer):
    def __init__(
        self,
        data: np.ndarray,
        results_dir: str,
        neurons_n=200,
        epsilon="auto",
        lambda_param="auto",
        plotting_colors: dict = NG_colors,
        **kwargs
    ) -> None:
        super().__init__(
            data, neurons_n, results_dir, plotting_colors=plotting_colors, **kwargs
        )
        self.epsilon = epsilon
        self.lambda_param = lambda_param

    def update(self, i: int, x: np.ndarray):
        distances = np.linalg.norm(self.neurons - x, axis=1)
        ranking = np.argsort(distances)

        for r, neuron_idx in enumerate(ranking):
            self.neurons[neuron_idx] += (
                self.epsilon
                * np.exp(-r / self.lambda_param)
                * (x - self.neurons[neuron_idx])
            )

        self.increase_age(ranking[0], ranking[1])
        self.remove_old_connections()
