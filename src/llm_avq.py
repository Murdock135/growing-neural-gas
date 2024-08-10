import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from tqdm import tqdm
import os
import imageio
import src.utils as utils
from datetime import datetime
from typing import TypeAlias

class AdaptiveVectorQuantizer:
    def __init__(
        self,
        data,
        neurons_n,
        lifetime,
        epsilon,
        _lambda,
        fig_save_path,
        max_local_iter='auto',
        global_iter=3,
    ):
        # Data
        self.data = data
        self.size = data.shape[0]

        # Neurons, Topology
        self.neurons = self.create_neurons(neurons_n)
        self.connection_matrix = np.zeros((self.neurons.shape[0], self.neurons.shape[0]))

        # Hyperparameters
        self.lifetime = lifetime
        self.e = epsilon
        self._lambda = _lambda

        # Path for saving results
        self.fig_save_path = fig_save_path
        if max_local_iter == 'auto':
            self.max_local_iter = self.size  # iterate over the data 3 times
        else:
            self.max_local_iter = max_local_iter

        # Global iteration
        self.global_iter = global_iter

        # Create subdirectories for each global iteration number
        for i in range(1, self.global_iter + 1):
            os.makedirs(os.path.join(self.fig_save_path, str(i)), exist_ok=True)

        # List for storing sampled datapoints
        self.sampled_datapoints = [[] for _ in range(self.global_iter)]

        # Attributes needed for functions that define the algorithm itself
        self.current_sample = None

    def run(self):
        print("Running Algorithm")

        # repeat the algorithm over the data for global iterations
        for j in tqdm(range(1, self.global_iter + 1)):
            # local iteration
            for i in tqdm(range(1, self.size + 1)):
                # sample without replacement
                while True:
                    choice = np.random.randint(0, self.size)
                    self.current_sample = self.data[choice]
                    if not utils.search_datapoint(self.sampled_datapoints[j - 1], self.current_sample.tolist()):
                        break

                # register sample in array of sampled datapoints
                self.sampled_datapoints[j - 1].append(self.current_sample.tolist())

                self.main_algorithm()

                # create plot and save it every 50th iteration
                if i % 50 == 0 or i == self.size:
                    self.save_plot(local_iter=i, global_iter=j, current_sample=self.current_sample)

            # Create GIF
            print("Run complete")
            self.create_gif(j)

    def main_algorithm(self):
        raise NotImplementedError("Subclasses should implement this!")

    def find_neighbors(self, r_index):
        '''Find the indexes of neighbors of neuron whose index is r_index (int)'''
        return np.nonzero(self.connection_matrix[r_index])[0]

    def alter_topology(self, r_index, c_index):
        '''Alter topology'''
        # if connection age < lifetime, increase age, else reset
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            self.connection_matrix[r_index, c_index] += 1  # changing the upper triangle only
        else:
            self.connection_matrix[r_index, c_index] = 0  # reset age

    def plot(self, local_iter, global_iter, current_sample, cmap=utils.cmap, neurons_color='k', data_color='#4575F3'):
        """This plots the data+neurons."""
        fig, ax = plt.subplots()

        # plot all data and all neurons
        ax.scatter(self.neurons[:, 0], self.neurons[:, 1], marker=".", label="neurons", c=neurons_color)  # plot neurons        
        ax.scatter(self.data[:, 0], self.data[:, 1], s=0.5, marker='o', label='data', c=data_color)  # plot data

        # convert the lists of sampled datapoints into np arrays
        arrays = [np.array(lst) for lst in self.sampled_datapoints]

        # Ensure length of cmap and arrays are 'compatible'
        assert cmap.N >= len(arrays), "The length of cmap must be at least the global iteration number"

        # plot sampled datapoints and collect scatter plots
        scatter_plots_of_sampled_datapoints = []
        for i, array in enumerate(arrays):
            if array.size > 0:
                scatter = ax.scatter(array[:, 0], array[:, 1], s=10, marker='o', facecolor=cmap(i))
                scatter_plots_of_sampled_datapoints.append(scatter)

        # plot current sample
        ax.scatter(current_sample[0], current_sample[1], s=50, marker='o', label='current sample', facecolor='green', edgecolor='k')

        # colorbar
        cmap = cmap.with_extremes(under=data_color, over='red')
        bounds = [0, 1, 2, 3]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, extend='both', spacing='proportional')
        cbar.set_label('Number of times sampled')

        # plot connections between neurons
        for r in range(self.connection_matrix.shape[0]):  # checking only upper triangle
            for c in range(self.connection_matrix.shape[1]):  # checking only upper triangle
                if self.connection_matrix[r, c] > 0:
                    neuron_a = self.neurons[r]
                    neuron_b = self.neurons[c]

                    x_coords = [neuron_a[0], neuron_b[0]]
                    y_coords = [neuron_a[1], neuron_b[1]]

                    line = mlines.Line2D(x_coords, y_coords, color=neurons_color)
                    ax.add_line(line)

        ax.legend()
        ax.set_title(f"Global iteration: {global_iter}\nLocal iteration: {local_iter}")
        return fig

    def save_plot(self, local_iter, global_iter, current_sample):
        # create figure
        fig = self.plot(local_iter, global_iter, current_sample)

        # save figure
        file_name = os.path.join(self.fig_save_path, str(global_iter), f"frame_{local_iter:04d}.png")
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

    def create_gif(self, global_iter):
        figures = []
        results_for_global_iter = os.path.join(self.fig_save_path, str(global_iter))
        filenames = os.listdir(results_for_global_iter)

        for f in filenames:
            filepath = os.path.join(self.fig_save_path, str(global_iter), f)
            figures.append(imageio.imread(filepath))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(self.fig_save_path, str(global_iter), f"neural_gas_animation_{timestamp}.gif")
        imageio.mimsave(gif_path, figures)

class NeuralGas(AdaptiveVectorQuantizer):
    def main_algorithm(self):
        # calculate distances
        distances = np.linalg.norm(self.neurons - self.current_sample, axis=1)

        # sort neurons by distance from input
        neuron_idx_sorted_by_dist = np.argsort(distances)

        # alter connection between closest neurons
        self.alter_topology(neuron_idx_sorted_by_dist[0], neuron_idx_sorted_by_dist[1])

        # update neurons
        for idx, _ in enumerate(self.neurons):
            # find # of better neurons for each neuron
            if idx == 0:
                k = utils.zero(neuron_idx_sorted_by_dist)[0].item()  # Note: only 1 element will be 0 in neuron_idx_sorted_by_dist. Thus, zero(neuron_idx_sorted_dist) returns a 1-tuple
            else:
                k = np.nonzero(neuron_idx_sorted_by_dist == idx)[0].item()

            # Update
            self.neurons[idx] += (self.e * np.exp(-k / self._lambda) * (self.current_sample - self.neurons[idx]))

class GrowingNeuralGas(AdaptiveVectorQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.errors = np.zeros(self.neurons.shape[0])

    def main_algorithm(self):
        # calculate distances
        distances = np.linalg.norm(self.neurons - self.current_sample, axis=1)

        # sort neurons by distance from input
        neuron_idx_sorted_by_dist = np.argsort(distances)

        # Best neuron
        best_neuron_idx = neuron_idx_sorted_by_dist[0]

