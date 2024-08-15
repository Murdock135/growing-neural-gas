import pandas as pd
import numpy as np
import torch.nn as nn

foo = np.array([1,2,4])

print(foo)

bar = pd.DataFrame(foo)

class myNet():
    def __init__(self, input, output) -> None:
        fc1 = nn.Linear(input, output)


print("bar: ", bar)

import matplotlib.pyplot as plt
import numpy as np

# Example data
data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
sample_counts = np.array([10, 15, 5, 20, 25])  # Example count data

# Normalize sample_counts to range [0, 1] for colormap
norm = plt.Normalize(sample_counts.min(), sample_counts.max())

# Create a colormap
cmap = plt.cm.viridis  # You can choose other colormaps as well

# Plot the data points, using sample_counts to set the color
plt.scatter(data[:, 0], data[:, 1], c=sample_counts, cmap=cmap, norm=norm)

# Add a colorbar to show the mapping
plt.colorbar(label='Sample Count')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points Colored by Sample Count')
plt.show()

class GrowingNeuralGas(BaseNeuralGas):
    def __init__(self, data, max_neurons=100, max_age=50, learning_rate_b=0.2, learning_rate_n=0.006, max_iterations=1000, epochs=1):
        super().__init__(data, num_neurons=2, learning_rate=learning_rate_b, max_iterations=max_iterations, epochs=epochs)
        self.max_neurons = max_neurons
        self.max_age = max_age
        self.learning_rate_n = learning_rate_n
        self.edges = np.zeros((2, 2))
        self.errors = np.zeros(2)

    def update_neurons(self, x):
        distances = np.linalg.norm(self.neurons - x, axis=1)
        s1, s2 = np.argsort(distances)[:2]

        self.errors[s1] += distances[s1]**2
        self.neurons[s1] += self.learning_rate * (x - self.neurons[s1])

        neighbors = np.where(self.edges[s1] > 0)[0]
        for n in neighbors:
            self.edges[s1, n] += 1
            self.edges[n, s1] += 1
            self.neurons[n] += self.learning_rate_n * (x - self.neurons[n])

        self.edges[s1, s2] = 0
        self.edges[s2, s1] = 0

    def train(self):
        plt.ion()  # Turn on interactive mode for real-time plotting
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for i in range(self.max_iterations):
                np.random.shuffle(self.data)  # Shuffle data once per iteration
                for idx, x in enumerate(self.data):
                    self.sampling_count[idx] += 1
                    self.update_neurons(x)
                    self.plot_neurons(x, idx)
                    plt.pause(0.01)  # Pause to update the plot
                self.add_new_neuron()
                self.remove_old_edges()

    def add_new_neuron(self):
        if len(self.neurons) < self.max_neurons:
            q = np.argmax(self.errors)
            neighbors = np.where(self.edges[q] > 0)[0]
            f = neighbors[np.argmax(self.errors[neighbors])]
            new_neuron = (self.neurons[q] + self.neurons[f]) / 2
            self.neurons = np.vstack([self.neurons, new_neuron])
            new_edges = np.zeros((len(self.neurons), len(self.neurons)))
            new_edges[:-1, :-1] = self.edges
            new_edges[q, -1] = 0
            new_edges[-1, q] = 0
            new_edges[f, -1] = 0
            new_edges[-1, f] = 0
            self.edges = new_edges
            self.errors = np.append(self.errors, 0)
            self.errors[q] *= 0.5
            self.errors[f] *= 0.5

    def remove_old_edges(self):
        self.edges[self.edges > self.max_age] = 0

# Example usage
data = np.random.rand(1000, 2)
gng = GrowingNeuralGas(data, max_neurons=50, max_iterations=100, epochs=5)
gng.train()
