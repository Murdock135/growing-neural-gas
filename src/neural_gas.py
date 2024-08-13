import numpy as np
from adaptive_vector_quantizer import AdaptiveVectorQuantizer


class NeuralGas(AdaptiveVectorQuantizer):
    def __init__(
        self,
        data: np.ndarray,
        results_dir: str,
        lifetime="auto",
        max_iter: int = "auto",
        epochs: int = 3,
        plot_interval: int = 100,
        sampling_without_replacement: bool = True,
        neurons_n=200,
        epsilon="auto",
        lambda_param="auto",
        **plotting_colors,
    ) -> None:
        super().__init__(
            data,
            neurons_n,
            results_dir,
            lifetime,
            max_iter,
            epochs,
            plot_interval,
            sampling_without_replacement,
            **plotting_colors,
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

        self.connect_neurons(ranking[0], ranking[1])
        self.remove_old_connections()
