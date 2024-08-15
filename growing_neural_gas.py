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
        max_neurons = 1000,
        eps_b="auto",
        eps_n="auto",
        lambda_param="auto",
        alpha='auto',
        decay='auto',
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
        self.max_neurons = max_neurons
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.decay = decay
        self.errors = np.zeros(neurons_n)

    def update(self, i: int, x: np.ndarray):
        distances = np.linalg.norm(self.neurons - x, axis=1)
        s1_idx, s2_idx = np.argsort(distances)[:2]
    
        neigbors_indices = np.nonzero(self.connection_matrix[s1_idx])[0]
        for n in neigbors_indices:
            self.increase_age(s1_idx, n)
            self.neurons[n] += self.eps_n * (x - self.neurons[n])

        self.errors[s1_idx] += distances[s1_idx] ** 2
        self.neurons[s1_idx] += self.eps_b * (x - self.neurons[s1_idx])

        if self.connection_matrix[s1_idx, s2_idx] > 0:
            self.connection_matrix[s1_idx, s2_idx] = 0
        else:
            self.connection_matrix[s1_idx, s2_idx] = 1

        self.remove_old_connections()

        if i % self.lambda_param == 0:
            self.insert_new_neuron()
    
    def delete_lonely_neurons(self):
        lonely_neuron_indices = np.nonzero(~np.any(self.connection_matrix, axis=1))[0]

        if len(lonely_neuron_indices) > 0:
            self.neurons = np.delete(self.neurons, lonely_neuron_indices, axis=0)
            self.connection_matrix = np.delete(self.connection_matrix, lonely_neuron_indices, axis=0)
            self.connection_matrix = np.delete(self.connection_matrix, lonely_neuron_indices, axis=1)
            self.errors = np.delete(self.errors, lonely_neuron_indices)

    def insert_new_neuron(self):
        if len(self.neurons) < self.max_neurons:
            # Create new neuron
            q_idx = np.argmax(self.errors)
            neigbors_indices = np.nonzero(self.connection_matrix[q_idx])[0]
            f_idx = neigbors_indices[np.argmax(self.errors[neigbors_indices])]
            new_neuron = 0.5 * (self.neurons[q_idx] + self.neurons[f_idx])
            self.neurons = np.vstack((self.neurons, new_neuron))

            # Adjust connection matrix
            self.connection_matrix = np.pad(self.connection_matrix, ((0, 1), (0, 1)))
            self.connection_matrix[q_idx, -1] = self.connection_matrix[-1, q_idx] = 1
            self.connection_matrix[f_idx, -1] = self.connection_matrix[-1, f_idx] = 1
            self.connection_matrix[q_idx, f_idx] = self.connection_matrix[f_idx, q_idx] = 0

            # Adjust errors
            self.errors[q_idx] *= self.alpha
            self.errors[f_idx] *= self.alpha
            self.errors = np.append(self.errors, self.errors[q_idx])

            # Decrease all errors
            self.errors *= self.decay

