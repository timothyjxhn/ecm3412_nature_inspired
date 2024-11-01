'''
BPP1: Bin Packing Problem 1
- 500 items
- weight of item i is i (e.g. item 1 has weight 1, item 2 has weight 2, etc.)
- 10 bins

BPP2: Bin Packing Problem 2
- 500 items
- weight of item i is (i^2)/2 (e.g. item 1 has weight 0.5, item 2 has weight 2, etc.)
- 50 bins
'''

import numpy as np
import random
import concurrent.futures

total_fitness_evals = 10000


class ACOBinPacker:
    def __init__(self, problem: int, ants: int, e_rate: float):
        """ Initialize ACOBinPacker instance
        
        Parameters:
            problem: problem number (1 or 2)
            ants: number of ants to use in 1 trial
            e_rate: evaporation rate
        """
        self.problem = problem
        if self.problem not in [1, 2]:
            raise ValueError("Invalid problem number")
        self.no_of_items = 500 if problem == 1 else 500
        self.no_of_bins = 10 if problem == 1 else 50
        self.e_rate = e_rate
        self.ants = ants
        self.p_matrix = self.initialize_pheromones()

    def run(self):
        """ Run the ACO algorithm to solve the Bin Packing Problem
        
        Returns:
            list: best path found by the algorithm
            fitness: fitness value of the best path
        """
        total_runs = total_fitness_evals // self.ants
        for i in range(total_runs):
            print(f"Run {i} of {total_runs}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.single_ant_run) for _ in range(self.ants)]
                runs = [future.result() for future in concurrent.futures.as_completed(futures)]
            for j in runs:
                path, fitness = j
                self.pheromone_update(path, fitness)
            self.apply_evaporation()
        best_path, best_fitness = self.single_ant_run()
        return best_path, best_fitness

    def single_ant_run(self):
        """ Run a single ant through the graph to find a path
        
        Returns:
            list: a path found by an ant
            fitness: fitness value of the path
        """
        path = self.generate_path()
        fitness = self.fitness_eval(path)
        return path, fitness

    def initialize_pheromones(self):
        """ Initialize and returns pheromone with random values
         
        Returns:
            matrix: 2D numpy array of size (no_of_items * no_of_bins) + 2
        """
        axis_len = (self.no_of_items * self.no_of_bins) + 1
        matrix = np.zeros((axis_len, axis_len))
        # initialize edge from start node to 1st item nodes
        matrix[0, 1:self.no_of_bins + 1] = np.random.random(self.no_of_bins)
        for i in range(1, axis_len - self.no_of_bins - 1, self.no_of_bins):
            start_node = i + self.no_of_bins
            matrix[i:start_node, start_node:start_node + self.no_of_bins] = np.random.random((self.no_of_bins, self.no_of_bins))
        return matrix
        
    def generate_path(self):
        """ Generate path for ants to traverse
         
        Returns:
            a list of node numbers (integers) representing the path
        """
        path = [0]
        current_node = 0
        for i in range(self.no_of_items):
            bin_choices = self.p_matrix[current_node]
            total_pheromone = sum(bin_choices)
            normalized_pheromones = [pheromone / total_pheromone for pheromone in bin_choices]
            selected_node = random.choices([i for i in range(len(bin_choices))], normalized_pheromones)[0]
            path.append(selected_node)
            current_node = selected_node
        return path
     
    def fitness_eval(self, path: list):
        """evaluate fitness by getting difference between heaviest and lightest bins
        
        Parameters:
            path: list of node integers
        
        Returns:
            Value of diff between heaviest and lightest bin 
        """
        bins = {}
        pair_set = [self.node_to_pair(node) for node in path]
        for pair in pair_set:
            weight, bin = pair
            weight = (weight ** 2) / 2 if self.problem == 2 else weight
            bins[bin] = bins.get(bin, 0) + weight
        bins = list(bins.values())
        return max(bins) - min(bins)
    
    def pheromone_update(self, path: list, fitness: int):
        """Update pheromones based on fitness of path
        
        Parameters:
            path: path of nodes chosen by ant; list of ints
            fitness: raw fitness value of path (i.e. difference between heaviest and lightest bin)
        """
        for i in range(1, len(path)):
            self.p_matrix[path[i - 1]][path[i]] = 100 / fitness if fitness != 0 else 100

    def apply_evaporation(self):
        """Apply evaporation to pheromone matrix in-place"""
        self.p_matrix *= self.e_rate

    def node_to_pair(self, node: int):
        """Convert node number to pair of item weight and bin number
        
        Parameters:
            node: node number
        
        Returns:
            tuple of item weight and bin number
        """
        if node == 0:
            return (0, 0)
        item = (node - 1) // self.no_of_items + 1
        bin = (node - 1) % self.no_of_items + 1
        return (item, bin)


def main():
    p = 10
    e = 0.9
    bpp = ACOBinPacker(1, p, e)
    best_path, best_fitness = bpp.run()
    print(f"Best Path: {best_path}")
    print(f"Best Fitness: {best_fitness}")


if __name__ == '__main__':
    main()
