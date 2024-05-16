import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class NeuralGas:
    def __init__(self, data, neurons_n, lifetime, epsilon, _lambda, max_iter):
        self.data = data
        self.neurons = self.create_neurons(neurons_n)
        self.connection_matrix = np.zeros((self.neurons.shape[0], self.neurons.shape[0]))
        self.lifetime = lifetime
        self.e = epsilon
        self._lambda = _lambda
        self.max_iter = max_iter

    def run(self):
        for iter in range(self.max_iter):
            print("iter: ", iter)
            choice = np.random.randint(0, self.data.shape[0])
            _input = self.data[choice]
            
            self.ng_one_input(_input)

    def ng_one_input(self, _input):
        # calculate distances
        distances = np.linalg.norm(self.neurons-_input, axis=1)

        # sort neurons by distance from input
        neuron_idx_sorted_by_dist = np.argsort(distances) # e.g. [0.34, 0.65, 0.1] -> [2, 0, 1]
        # sorted_neurons = self.neurons[neuron_idx_sorted_by_dist]

        # make connection between two closest neurons
        self.alter_connection(neuron_idx_sorted_by_dist[0], neuron_idx_sorted_by_dist[1])

        # plot after making connection
        self.plot(_input)

        # update neurons
        for idx, _ in enumerate(self.neurons):
            # print("neuron = ", _)
            # find # of better neurons for each neuron
            k = np.where(neuron_idx_sorted_by_dist==idx)[0][0]
            
            # update
            self.neurons[idx] += self.e * np.exp(-k/self._lambda) * (_input - self.neurons[idx])

            # # plot after change
            # self.plot(_input, self.neurons[idx])

    def alter_connection(self, r_index, c_index):
        # check if connection exists
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            self.connection_matrix[r_index, c_index] += 1 # changing the upper triange only
        else:
            self.connection_matrix[r_index, c_index] = 0
    
    def plot(self, *points):
        fig, ax = plt.subplots()

        ax.scatter(self.neurons[:,0], self.neurons[:,1], marker='X', label='neurons')
        ax.scatter(self.data[:,0], self.data[:,1], s=0.5, label='data')

        if points:
            ax.scatter(points[0][0], points[0][1], label='current input')

            if len(points) > 1:
                ax.scatter(points[1][0], points[1][1], marker='X', label='neuron')

        # find connections in connection matrix
        for r in range(self.connection_matrix.shape[0]):
            for c in range(self.connection_matrix.shape[1]):
                if self.connection_matrix[r,c] > 0:
                    neuron_a = self.neurons[r]
                    neuron_b = self.neurons[c]

                    x_coords = [neuron_a[0], neuron_b[0]]
                    y_coords = [neuron_a[1], neuron_b[1]]

                    line = mlines.Line2D(x_coords, y_coords)
                    ax.add_line(line)                   

        plt.legend()
        plt.show()

    def create_neurons(self, neurons_n, dist='uniform'):
        x_min, y_min = np.amin(self.data, axis=0)
        x_max, y_max = np.amax(self.data, axis=0)

        if dist == 'uniform':
            x_coords = np.random.uniform(x_min, x_max, neurons_n)
            y_coords = np.random.uniform(y_min, y_max, neurons_n)
        
        # create else block for when dist == 'normal' later

        neurons = np.column_stack((x_coords, y_coords))
        return neurons
