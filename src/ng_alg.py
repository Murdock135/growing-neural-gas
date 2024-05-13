import numpy as np
import matplotlib.pyplot as plt

class NeuralGas:
    def __init__(self, neurons, data, lifetime, epsilon, _lambda):
        self.neurons = neurons
        self.data = data
        self.connection_matrix = np.zeros((neurons.shape[0], neurons.shape[0]))
        self.lifetime = lifetime
        self.e = epsilon
        self._lambda = _lambda

    def ng_one_input(self, single_input):

        # calculate distances
        distances = np.linalg.norm(self.neurons-single_input, axis=1)

        # sort neurons by distance from input
        neuron_idx_sorted_by_dist = np.argsort(distances) # e.g. [0.34, 0.65, 0.1] -> [2, 0, 1]
        # sorted_neurons = self.neurons[neuron_idx_sorted_by_dist]

        # make connection between two closest neurons
        self.alter_connection(neuron_idx_sorted_by_dist[0], neuron_idx_sorted_by_dist[1])

        # update neurons
        for idx, neuron in enumerate(self.neurons):
            # find # of better neurons for each neuron
            k = np.where(neuron_idx_sorted_by_dist==idx)[0][0]
            
            neuron = neuron + self.e * np.exp(-k/self._lambda) * (single_input - neuron)

    def alter_connection(self, r_index, c_index):
        # check if connection exists
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            self.connection_matrix[r_index, c_index] += 1 # changing the upper triange only
        else:
            self.connection_matrix[r_index, c_index] = 0
    
    def plot(self):
        plt.scatter(self.neurons[:,0], self.neurons[:,1], label='neurons')
        plt.scatter(self.data[:,0], self.data[:,1], label='data')

        plt.legend()
        plt.show()
