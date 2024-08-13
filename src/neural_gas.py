import numpy as np
from adaptive_vector_quantizer import AdaptiveVectorQuantizer


class NeuralGas(AdaptiveVectorQuantizer):
    def __init__(
        self,
        data: np.ndarray,
        results_dir: str,
        max_iter: int = "auto",
        epochs: int = 3,
        plot_interval: int = 100,
        sampling_without_replacement: bool = True,
        neurons_n=200,
        epsilon="auto",
        lambda_param="auto",
        lifetime="auto",
        **plotting_colors,
    ) -> None:
        super().__init__(
            data,
            results_dir,
            max_iter,
            epochs,
            plot_interval,
            sampling_without_replacement,
            **plotting_colors,
        )
        self.neurons_n = neurons_n
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.lifetime = lifetime

    def run(self):
        shuffled_data: np.ndarray = self.shuffle_data()
        if self.max_iter == 'auto':
            assert self.max_iter == self.get_data_size(), "Max iterations is set to 'auto'. Data size need to equal number of iterations (max_iter)"
        for epoch in range(self.epochs):
            for i in range(self.max_iter):
                pass
                
