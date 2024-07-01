import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tqdm
import os
import imageio
import datetime

class NeuralGas:
    def __init__(self, data:list, neurons_n, lifetime, epsilon, _lambda, max_iter, fig_save_path):
        self.data = data
        self.neurons = self.create_neurons(neurons_n)
        self.connection_matrix = np.zeros((self.neurons.shape[0], self.neurons.shape[0]))
        self.lifetime = lifetime
        self.e = epsilon
        self._lambda = _lambda
        self.max_iter = max_iter
        self.fig_save_path = fig_save_path

    def run(self):
        for iter in tqdm(range(self.max_iter)):
            choice = np.random.randint(0, self.data.shape[0])
            _input = self.data[choice]
            self.ng_one_input(_input)
            self.save_plot(iter,_input)

        print("Run complete")
        self.create_gif()

    def ng_one_input(self, _input):
        # calculate distances
        distances = np.linalg.norm(self.neurons-_input, axis=1)

        # sort neurons by distance from input
        neuron_idx_sorted_by_dist = np.argsort(distances) # e.g. [0.34, 0.65, 0.1] -> [2, 0, 1]
        # sorted_neurons = self.neurons[neuron_idx_sorted_by_dist]

        # alter connection between closest neurons
        self.alter_connection(neuron_idx_sorted_by_dist[0], neuron_idx_sorted_by_dist[1])

        # update neurons
        for idx, _ in enumerate(self.neurons):
            # find # of better neurons for each neuron
            if idx == 0:
                k = zero(neuron_idx_sorted_by_dist)[0].item() # Note: only 1 element will be 0 in neuron_idx_sorted_by_dist. Thus, zero(neuron_idx_sorted_dist) returns a 1-tuple
            else:
                k = np.nonzero(neuron_idx_sorted_by_dist == idx)[0].item()
            
            # update
            self.neurons[idx] += self.e * np.exp(-k/self._lambda) * (_input - self.neurons[idx])

    def alter_connection(self, r_index, c_index):
        # print(f"lifetime: {self.connection_matrix[r_index, c_index]}")

        # if connection age < lifetime, increase age, else reset
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            # print(f"lifetime between neurons {r_index} and {c_index}=", self.connection_matrix[r_index, c_index])
            self.connection_matrix[r_index, c_index] += 1 # changing the upper triange only
        else:
            print('lifetime reached. Removing connection')
            self.connection_matrix[r_index, c_index] = 0 # reset age
    
    def plot(self, iter, *args):
        '''This plots the data+neurons.
        Info about *args:
            args is a list containing the coordinates of the current input and current neuron ('current' as in the one sampled in the current iteration of the algorithm) thus it is
            [current_input, current_neuron]'''
        fig, ax = plt.subplots()

        ax.scatter(self.neurons[:,0], self.neurons[:,1], marker='X', label='neurons') # plot neurons
        ax.scatter(self.data[:,0], self.data[:,1], s=0.5, label='data') # plot data

        if args:
            cur_input = args[0]
            ax.scatter(cur_input[0], cur_input[1], label='current input')

            if len(args) > 1:
                # optionally, point out the neuron being updated
                neuron = args[1]
                ax.scatter(neuron[0], neuron[1], marker='X', label='neuron')

        # find connections in connection matrix
        for r in range(self.connection_matrix.shape[0]): # checking only upper triangle
            for c in range(self.connection_matrix.shape[1]): # checking only upper triange
                if self.connection_matrix[r,c] > 0:
                    neuron_a = self.neurons[r]
                    neuron_b = self.neurons[c]

                    x_coords = [neuron_a[0], neuron_b[0]]
                    y_coords = [neuron_a[1], neuron_b[1]]

                    line = mlines.Line2D(x_coords, y_coords)
                    ax.add_line(line)                   

        ax.legend()
        ax.set_title(f"Iteration {iter}")
        return fig
    
    def save_plot(self, iter, *points):
        fig = self.plot(iter, *points)
        file_name = os.path.join(self.fig_save_path, f"frame_{iter:04d}.png")
        fig.savefig(file_name)
        plt.close(fig)


    def create_neurons(self, neurons_n, dist='uniform'):
        x_min, y_min = np.amin(self.data, axis=0)
        x_max, y_max = np.amax(self.data, axis=0)

        if dist == 'uniform':
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

