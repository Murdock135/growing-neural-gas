# growing_neural_gas.py

import numpy as np
from adaptive_vector_quantizer import AdaptiveVectorQuantizer
from typing import List


class GrowingNeuralGas(AdaptiveVectorQuantizer):
    def __init__(
        self,
        data: np.ndarray,
        fig_save_path: str,
        neurons_n: int,
        max_iterations: int,
        global_iterations: int,
        plot_interval: int,
        eps_b: float,
        eps_n: float,
        lifetime: int,
        alpha: float,
        decay: float,
        lambda_param: int,
    ) -> None:
        """
        Initialize the Growing Neural Gas algorithm.

        Args:
            data (np.ndarray): Input data points.
            fig_save_path (str): Path to save output figures.
            neurons_n (int): Maximum number of neurons.
            max_iterations (int): Maximum number of iterations per global iteration.
            global_iterations (int): Number of global iterations.
            plot_interval (int): Interval for plotting and saving figures.
            eps_b (float): Learning rate for the best matching unit.
            eps_n (float): Learning rate for the neighbors of the best matching unit.
            lifetime (int): Maximum age for connections between neurons.
            alpha (float): Error reduction factor for winner neuron.
            decay (float): General error reduction factor.
            lambda_param (int): Interval for inserting new neurons.
        """
        super().__init__(
            data,
            fig_save_path,
            neurons_n,
            max_iterations,
            global_iterations,
            plot_interval,
        )
        self.eps_b: float = eps_b
        self.eps_n: float = eps_n
        self.lifetime: int = lifetime
        self.alpha: float = alpha
        self.decay: float = decay
        self.lambda_param: int = lambda_param
        self.errors: np.ndarray = np.zeros(self.neurons_n)

    def _update(self) -> None:
        """Update the Growing Neural Gas network for one iteration."""
        # Calculate distances between all neurons and current sample
        current_sample = self.data[self.current_sample_idx]
        distances = np.linalg.norm(self.neuron_positions - current_sample, axis=1)
        neuron_idx_sorted_by_dist = np.argsort(distances)
        best_neuron_idx = neuron_idx_sorted_by_dist[0]  # Best unit, 'unit s1'

        # Increment the age of all edges emanating from s1
        neigbor_idxs = self.find_neighbors(best_neuron_idx)
        self.connection_matrix[best_neuron_idx, neigbor_idxs] += 1
        self.connection_matrix[neigbor_idxs, best_neuron_idx] += 1

        # Add the squared distance between input signal and nearest unit to a local counter variable
        self.errors[best_neuron_idx] += distances[best_neuron_idx] ** 2

        # Move the best neuron and its topological neighbors towards input signal by fractions of the total distance
        self.neuron_positions[best_neuron_idx] += self.eps_b * (
            current_sample - self.neuron_positions[best_neuron_idx]
        )
        self.neuron_positions[neigbor_idxs] += self.eps_n * (
            current_sample - self.neuron_positions[neigbor_idxs]
        )

        # If best neuron, s1, and second best neuron, s2 are connected, increase age, else create it
        self.alter_topology(best_neuron_idx, neuron_idx_sorted_by_dist[1])

        # Remove old connections and lonely neurons
        self.remove_old_connections()

        # If number of sampled points > lambda, insert new neuron
        if (
            len(self.sampled_indices[self.current_epoch - 1])
            % self.lambda_param
            == 0
        ):
            self.insert_new_neuron()

    def find_neighbors(self, r_index: int) -> np.ndarray:
        """
        Find the indices of neighbors for a given neuron.

        Args:
            r_index (int): Index of the neuron.

        Returns:
            np.ndarray: Indices of neighboring neurons.
        """
        return np.nonzero(self.connection_matrix[r_index])[0]

    def delete_lonely_neurons(self) -> None:
        """Delete neurons without any connections."""
        lonely_neuron_idxs = np.nonzero(np.all(self.connection_matrix == 0, axis=1))[0]
        if len(lonely_neuron_idxs) > 0:
            self.neuron_positions = np.delete(self.neuron_positions, lonely_neuron_idxs, axis=0)
            self.connection_matrix = np.delete(self.connection_matrix, lonely_neuron_idxs, axis=0)
            self.connection_matrix = np.delete(self.connection_matrix, lonely_neuron_idxs, axis=1)
            self.errors = np.delete(self.errors, lonely_neuron_idxs)

    def insert_new_neuron(self) -> None:
        """Insert a new neuron between the neuron with the highest and second highest errors."""
        errors_sorted_descending = np.argsort(self.errors)[::-1]
        q_idx, f_idx = errors_sorted_descending[0], errors_sorted_descending[1]
        new_position = 0.5 * (self.neuron_positions[q_idx] + self.neuron_positions[f_idx])
        self.neuron_positions = np.vstack((self.neuron_positions, new_position))

        # Adjust connection matrix
        self.connection_matrix = np.pad(self.connection_matrix, ((0, 1), (0, 1)))
        self.connection_matrix[q_idx, -1] = self.connection_matrix[-1, q_idx] = 1
        self.connection_matrix[f_idx, -1] = self.connection_matrix[-1, f_idx] = 1

        # Adjust errors
        self.errors[q_idx] *= self.alpha
        self.errors[f_idx] *= self.alpha
        self.errors = np.append(self.errors, self.errors[q_idx])

        # Decrease all errors by some factor 'decay'
        self.errors *= self.decay

    def alter_topology(self, r_index: int, c_index: int) -> None:
        """
        Update the connection between two neurons.

        Args:
            r_index (int): Index of the first neuron.
            c_index (int): Index of the second neuron.
        """
        if self.connection_matrix[r_index, c_index] == 0:
            # Create new connection
            self.connection_matrix[r_index, c_index] = self.connection_matrix[c_index, r_index] = 1
        else:
            # Reset age of existing connection
            self.connection_matrix[r_index, c_index] = self.connection_matrix[c_index, r_index] = 1

    def remove_old_connections(self) -> None:
        """Remove connections older than the specified lifetime and delete lonely neurons."""
        old_connections = self.connection_matrix > self.lifetime
        self.connection_matrix[old_connections] = 0
        self.delete_lonely_neurons()