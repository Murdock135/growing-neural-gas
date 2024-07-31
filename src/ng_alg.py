import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
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
        max_local_iter='auto',
        global_iter = 3
    ):
        self.data = data
        self.size = data.shape[0]
        self.neurons = self.create_neurons(neurons_n)
        self.connection_matrix = np.zeros((self.neurons.shape[0], self.neurons.shape[0]))
        self.lifetime = lifetime
        self.e = epsilon
        self._lambda = _lambda
        self.fig_save_path = fig_save_path
        if max_local_iter == 'auto':
            self.max_local_iter = self.size # iterate over the data 3 times

        self.global_iter = global_iter

        # List for lists for storing sampled datapoints
        self.sampled_datapoints = []

        for _ in range(self.global_iter):
            self.sampled_datapoints.append([])


    def run(self):
        print("Running Neural Gas")

        # repeat the algorithm over the data 3 times ('global iteration')
        for j in tqdm(range(1, self.global_iter+1)):
            
            # local iteration
            for i in tqdm(range(1, self.size+1)): 

                # sample without replacement
                choice = np.random.randint(0, self.size)
                sample = self.data[choice]

                while utils.check_if_in_arr(self.sampled_datapoints[j-1], sample):
                    choice = np.random.randint(0, self.size)
                    sample = self.data[choice]

                # register sample in array of sampled datapoints
                self.sampled_datapoints[j-1].append(sample)

                self.algorithm(sample)
                
                # create plot and save it every 50th iteration
                if i % 50 == 0 or i == self.size:
                    self.save_plot(local_iter=i,
                               global_iter=j,
                               current_sample=sample)

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
            # print("lifetime reached. Removing connection")
            self.connection_matrix[r_index, c_index] = 0  # reset age
    
    def plot(self, local_iter, global_iter, current_sample, cmap=utils.cmap, neurons_color='k', data_color='blue'):
        """This plots the data+neurons."""
        fig, ax = plt.subplots()

        # plot all data and all neurons
        ax.scatter(self.neurons[:, 0], self.neurons[:, 1], marker=".", label="neurons", c=neurons_color)  # plot neurons        
        ax.scatter(self.data[:, 0], self.data[:, 1], s=0.5, marker='o', label='data', c=data_color) # plot data

        
        # convert the lists of sampled datapoints into np arrays
        arrays = [np.array(list) for list in self.sampled_datapoints]

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
        bounds = [0,1,2,3]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sm = mpl.cm.ScalarMappable(cmap=cmap,norm=norm)
        cbar = fig.colorbar(sm, ax=ax, extend='both', spacing='proportional')
        cbar.set_label('Number of times sampled')

        # plot connections between neurons
        for r in range(self.connection_matrix.shape[0]):  # checking only upper triangle
            for c in range(self.connection_matrix.shape[1]):  # checking only upper triange
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

        # make directory for separate global iteration
        dir_path = os.path.join(self.fig_save_path, str(global_iter))
        os.makedirs(dir_path, exist_ok=True)

        # save figure
        file_name = os.path.join(dir_path, f"frame_{local_iter:04d}.png")
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

        for iter in range(self.max_local_iter):
            file_name = os.path.join(self.fig_save_path, f"frame_{iter:04d}.png")
            figures.append(imageio.imread(file_name))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(self.fig_save_path, f"neural_gas_animation_{timestamp}.gif")
        imageio.mimsave(gif_path, figures)
    
    def reset(self):
        pass
