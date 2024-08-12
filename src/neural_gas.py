import numpy as np
from adaptive_vector_quantizer import AdaptiveVectorQuantizer


class NeuralGas(AdaptiveVectorQuantizer):
    def __init__(
        self,
        data: np.ndarray,
        fig_save_path: str,
        max_iterations: int,
        max_epochs: int,
        plot_interval: int,
        sample_without_replacement: bool,
        neurons_n: int,
        epsilon: float,
        lambda_param: float,
        lifetime: int,
        **kwargs
    ) -> None:
        """
        Initialize the Neural Gas algorithm.

        Args:
            data (np.ndarray): Input data points.
            fig_save_path (str): Path to save output figures.
            neurons_n (int): Number of neurons.
            max_iterations (int): Maximum number of iterations per global iteration.
            global_iterations (int): Number of global iterations.
            plot_interval (int): Interval for plotting and saving figures.
            epsilon (float): Learning rate.
            lambda_param (float): Decay constant for learning rate.
            lifetime (int): Maximum age for connections between neurons.

        Keyword Args:
            neuron_dist (str): The type of distribution from which neurons
                should be initialized into the data-space.
        """
        super().__init__(
            data,
            fig_save_path,
            max_iterations,
            max_epochs,
            plot_interval,
            sample_without_replacement
        )
        self.neurons_n: int = neurons_n
        self.neuron_dist: str = ('uniform' if not kwargs else kwargs['neuron_dist'].lower())
        self.neuron_positions: np.ndarray = self.create_neurons(self.neurons_n)
        self.connection_matrix: np.ndarray = np.zeros((self.neurons_n, self.neurons_n))

        self.epsilon: float = epsilon
        self.lambda_param: float = lambda_param
        self.lifetime: int = lifetime

    def _update(self) -> None:
        """Update the Neural Gas network for one iteration."""
        # Calculate distances between all neurons and the current sample
        current_sample = self.data[self.current_sample_idx]
        distances = np.linalg.norm(self.neuron_positions - current_sample, axis=1)

        # Sort neurons by distance to the current sample
        neuron_idx_sorted_by_dist = np.argsort(distances)

        # Update the topology (connections between neurons)
        self.alter_topology(neuron_idx_sorted_by_dist[0], neuron_idx_sorted_by_dist[1])

        # Update each neuron's position
        for idx, k in enumerate(np.argsort(neuron_idx_sorted_by_dist)):
            update = (
                self.epsilon
                * np.exp(-k / self.lambda_param)
                * (current_sample - self.neuron_positions[idx])
            )
            self.neuron_positions[idx] += update

    def alter_topology(self, r_index: int, c_index: int) -> None:
        """
        Update the connection between two neurons.

        Args:
            r_index (int): Index of the first neuron.
            c_index (int): Index of the second neuron.
        """
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            # Increment the age of the connection
            self.connection_matrix[r_index, c_index] += 1
            self.connection_matrix[c_index, r_index] += 1
        else:
            # If the connection is too old, remove it
            self.connection_matrix[r_index, c_index] = self.connection_matrix[c_index, r_index] = 0
    
    def create_neurons(self) -> np.ndarray:
        """
        Initializes the positions of neurons in the Growing Neural Gas (GNG) algorithm.

        This function generates the initial positions of the neurons within the space 
        spanned by the input data. The neurons' positions are determined based on the 
        specified distribution, which can be either uniform or normal. The function 
        supports arbitrary-dimensional data, ensuring that neurons are distributed 
        appropriately in all dimensions of the input space.

        Returns:
            np.ndarray: An array of shape `(self.neurons_n, self.dim)` containing the 
            initialized positions of the neurons.

        Raises:
            ValueError: If an unsupported distribution type is specified in `self.neuron_dist`.

        Distribution types:
        - "uniform": Neurons are initialized uniformly within the bounding box of the input data.
        - "normal": Neurons are initialized according to a normal distribution, with mean 
        and standard deviation computed from the input data.

        Attributes:
            self.data (np.ndarray): The input data used to define the space in which the neurons 
                                    will be initialized. Shape is `(n_samples, n_features)`.
            self.neurons_n (int): The number of neurons to initialize.
            self.dim (int): The dimensionality of the input data, i.e., the number of features.
            self.neuron_dist (str): The type of distribution used to initialize the neurons, 
                                    either "uniform" or "normal".
        """        
        min_values = np.amin(self.data, axis=0)
        max_values = np.amax(self.data, axis=0)

        rng = np.random.default_rng(0)
        if self.neuron_dist == "uniform":
            return rng.uniform(
                low=min_values, high=max_values, size=(self.neurons_n, self.dim)
            )
        elif self.neuron_dist == "normal":
            mean = np.mean(self.data, axis=0)
            std = np.std(self.data, axis=0)
            return rng.normal(mean, std, size=(self.neurons_n, self.dim))
        else:
            raise ValueError(f"Unsupported distribution: {self.neuron_dist}")
