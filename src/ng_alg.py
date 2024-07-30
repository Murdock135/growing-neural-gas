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
            self.max_local_iter = 3*self.data.shape[0] # iterate over the data 3 times

        self.global_iter = global_iter

        # Storing the number of times each datapoint has been sampled
        self.sample_counts = np.zeros(self.size, dtype=int)
        
    def run(self):
        print("Running Neural Gas")

        # repeat the algorithm over the data 3 times ('global iteration')
        for j in tqdm(range(1, self.global_iter+1)):
            
            # local iteration
            for i in tqdm(1, range(self.size+1)): 

                # sample without replacement
                choice = np.random.randint(0, self.data.shape[0])
                sample = self.data[choice]

                while self.sample_counts[choice] > j:
                    choice = np.random.randint(0, self.data.shape[0])
                    sample = self.data[choice]

                self.sample_counts[choice] += 1

                self.algorithm(sample)
                
                # create plot and save it every 50th iteration
                if i % 50 == 0:
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
            print("lifetime reached. Removing connection")
            self.connection_matrix[r_index, c_index] = 0  # reset age
    
    def plot(self, local_iter, global_iter, current_sample, cmap=utils.cmap):
        """This plots the data+neurons."""
        fig, ax = plt.subplots()

        # plot all data and all neurons
        ax.scatter(self.neurons[:, 0], self.neurons[:, 1], marker=".", label="neurons", c='0.3')  # plot neurons
        # plot data        
        scatter = ax.scatter(self.data[:, 0], self.data[:, 1], s=0.5, marker='o', label='data', c=self.sample_counts, cmap=cmap)

        # plot current sample
        ax.scatter(current_sample[0], current_sample[1], s=50, marker='o', label='current sample', facecolor='green', edgecolor='k')

        # colorbar
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(self.global_iter))
        cbar.set_label('Number of times sampled')

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
