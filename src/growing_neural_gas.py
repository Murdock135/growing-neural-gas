import numpy as np
from adaptive_vector_quantizer import AdaptiveVectorQuantizer


class GrowingNeuralGas(AdaptiveVectorQuantizer):
    def __init__(
        self,
        data: np.ndarray,
        results_dir: str,
        lifetime="auto",
        max_iter: int = "auto",
        epochs: int = 3,
        plot_interval: int = 100,
        sampling_without_replacement: bool = True,
        neurons_n=2,
        eps_b="auto",
        eps_n="auto",
        lambda_param="auto",
        **plotting_colors,
    ) -> None:
        super().__init__(
            data=data,
            neurons_n=neurons_n,
            results_dir=results_dir,
            lifetime=lifetime,
            max_iter=max_iter,
            epochs=epochs,
            plot_interval=plot_interval,
            sampling_without_replacement=sampling_without_replacement,
            **plotting_colors,
        )
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.lambda_param = lambda_param
        self.errors = np.zeros(neurons_n)

    def update(self, i: int, x: np.ndarray):
        distances = np.linalg.norm(self.neurons - x, axis=1)
        s1_idx, s2_idx = np.argsort(distances)[:2]

        self.errors[s1_idx] += distances[s1_idx] ** 2
        self.neurons[s1_idx] += self.eps_b * (x - self.neurons[s1_idx])
    
        neigbors_indices = np.nonzero(self.connection_matrix[s1_idx])[0]
        for n in neigbors_indices:
            self.connect_neurons(s1_idx, n)
            self.neurons[n] += self.eps_n * (x - self.neurons[n])
        
        self.remove_old_connections()
    
    def delete_lonely_neurons(self):
        pass

    def insert_new_neuron(self):
        pass
