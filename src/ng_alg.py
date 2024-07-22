import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from tqdm import tqdm
import os
import imageio
import src.utils as utils
from datetime import datetime
from typing import TypeAlias

# Aliases
Array2D: TypeAlias = np.ndarray

class NeuralGas:
    def __init__(
        self,
        data,
        neurons_n,
        lifetime,
        epsilon,
        _lambda,
        fig_save_path,
        max_iter='auto'
    ):
        self.data = data
        self.size = data.shape[0]
        self.neurons = self.create_neurons(neurons_n)
        self.connection_matrix = np.zeros((self.neurons.shape[0], self.neurons.shape[0]))
        self.lifetime = lifetime
        self.e = epsilon
        self._lambda = _lambda
        self.fig_save_path = fig_save_path
        if max_iter == 'auto':
            self.max_iter = 3*self.data.shape[0] # iterate over the data 3 times

        # array for storing the number of times each datapoint has been sampled
        self.sampled_n = np.zeros(shape=self.size)

    def run(self):
        print("Running Neural Gas")

        # repeat the algorithm over the data 3 times
        for j in tqdm(range(3)):
            # empty the bucket of previously sampled points
            prev_samples = np.empty((0, self.data.shape[1]))

            for i in tqdm(range(self.max_iter)): 

                # choose sample without replacement
                choice = np.random.randint(0, self.data.shape[0])
                while utils.check_if_in_arr(prev_samples, sample):
                    choice = np.random.randint(0, self.data.shape[0])

                sample = self.data[choice]
                self.sampled_n[choice] += 1

                self.algorithm(sample)
                self.save_plot(iter=i,
                               current_sample=sample,
                               prev_samples=prev_samples)

                # store the sample
                prev_samples = np.vstack([prev_samples, sample])

        print("Run complete")
        self.create_gif()

    def algorithm(self, sample):
        # calculate distances
        distances = np.linalg.norm(self.neurons - sample, axis=1)

        # sort neurons by distance from input
        neuron_idx_sorted_by_dist = np.argsort(
            distances
        )  # e.g. [0.34, 0.65, 0.1] -> [2, 0, 1]
        # sorted_neurons = self.neurons[neuron_idx_sorted_by_dist]

        # alter connection between closest neurons
        self.alter_connection(
            neuron_idx_sorted_by_dist[0], neuron_idx_sorted_by_dist[1]
        )

        # update neurons
        for idx, _ in enumerate(self.neurons):
            # find # of better neurons for each neuron
            if idx == 0:
                k = utils.zero(
                    neuron_idx_sorted_by_dist)[0].item()  # Note: only 1 element will be 0 in neuron_idx_sorted_by_dist. Thus, zero(neuron_idx_sorted_dist) returns a 1-tuple
            else:
                k = np.nonzero(neuron_idx_sorted_by_dist == idx)[0].item()

            # update
            self.neurons[idx] += (self.e * np.exp(-k / self._lambda) * (sample - self.neurons[idx]))

    def alter_connection(self, r_index, c_index):
        # print(f"lifetime: {self.connection_matrix[r_index, c_index]}")

        # if connection age < lifetime, increase age, else reset
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            # print(f"lifetime between neurons {r_index} and {c_index}=", self.connection_matrix[r_index, c_index])
            self.connection_matrix[r_index, c_index] += 1  # changing the upper triange only
        else:
            print("lifetime reached. Removing connection")
            self.connection_matrix[r_index, c_index] = 0  # reset age

    def plot(self, iter, current_sample, prev_samples: Array2D):
        """This plots the data+neurons.
        Info about **points:
            A dict containing the coordinates of the current neuron, current sampled input, and all inputs previously sampled thus it is
            {cur_neuron: nd.array, cur_sample: nd_array, prev_samples: nd_array}"""
        fig, ax = plt.subplots()

        # plot all data and all neurons
        ax.scatter(self.neurons[:, 0], self.neurons[:, 1], marker="o", label="neurons", c='orange')  # plot neurons
        ax.scatter(self.data[:, 0], self.data[:, 1], s=0.5, label="data", marker='o', c='0.8')  # plot data

        # plot current sample
        ax.scatter(current_sample[0], current_sample[1], marker="x", c='k', label="neuron")

        # plot previously sampled points
        if prev_samples.shape[0] > 0:
            ax.scatter(prev_samples[:, 0], prev_samples[:, 1], marker = 'p', c='cyan', label='previously picked samples')

        # plot connections between neurons
        for r in range(self.connection_matrix.shape[0]):  # checking only upper triangle
            for c in range(self.connection_matrix.shape[1]):  # checking only upper triange
                if self.connection_matrix[r, c] > 0:
                    neuron_a = self.neurons[r]
                    neuron_b = self.neurons[c]

                    x_coords = [neuron_a[0], neuron_b[0]]
                    y_coords = [neuron_a[1], neuron_b[1]]

                    line = mlines.Line2D(x_coords, y_coords)
                    ax.add_line(line)

        ax.legend()
        ax.set_title(f"Iteration {iter}")
        return fig

    def save_plot(self, iter, current_sample, prev_samples):
        fig = self.plot(iter, current_sample, prev_samples)
        file_name = os.path.join(self.fig_save_path, f"frame_{iter:04d}.png")
        fig.savefig(file_name)
        plt.close(fig)

    def create_neurons(self, neurons_n, dist="uniform"):
        x_min, y_min = np.amin(self.data, axis=0)
        x_max, y_max = np.amax(self.data, axis=0)

        if dist == "uniform":
            x_coords = np.random.uniform(x_min, x_max, neurons_n)
            y_coords = np.random.uniform(y_min, y_max, neurons_n)

        # create else block for when dist == 'normal' later

        neurons = np.column_stack((x_coords, y_coords))
        return neurons

    def create_gif(self):
        figures = []

        for iter in range(self.max_iter):
            file_name = os.path.join(self.fig_save_path, f"frame_{iter:04d}.png")
            figures.append(imageio.imread(file_name))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(self.fig_save_path, f"neural_gas_animation_{timestamp}.gif")
        imageio.mimsave(gif_path, figures)
    
    def reset(self):
        pass
