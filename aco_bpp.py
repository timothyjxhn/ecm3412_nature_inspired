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

import argparse
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
        self.best_fitness = np.inf
        self.worst_fitness = np.inf
        self.avg_fitness = np.inf

    def run(self):
        """ Run the ACO algorithm to solve the Bin Packing Problem
        
        Returns:
            list: best path found by the algorithm
            fitness: fitness value of the best path
        """
        total_runs = total_fitness_evals // self.ants
        all_fitness = []
        for _ in range(total_runs):
            # run ants concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.single_ant_run) for _ in range(self.ants)]
                runs = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # update pheromones and apply evaporation sequentially after all ants have run
            for j in runs:
                path, fitness = j
                all_fitness.append(fitness)
                self.pheromone_update(path, fitness)
            self.apply_evaporation()
        self.best_fitness = min(all_fitness)
        self.worst_fitness = max(all_fitness)
        self.avg_fitness = sum(all_fitness) / len(all_fitness)

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
        matrix[0, 1:self.no_of_bins + 1] = np.random.random(self.no_of_bins) # initialize edge from start node to 1st item nodes
        for x in range(1, axis_len - self.no_of_bins - 1, self.no_of_bins):
            y = x + self.no_of_bins # x, y is xy coordinate of sub-matrix corner to start from
            matrix[x:y, y:y + self.no_of_bins] = np.random.random((self.no_of_bins, self.no_of_bins))
        return matrix
        
    def generate_path(self):
        """ Generate path for ants to traverse
         
        Returns:
            a list of node numbers (integers) representing the path
        """
        path = [0]
        current_node = 0
        for _ in range(self.no_of_items):
            bin_choices = self.p_matrix[current_node] # refer to row of current node to find next bin choices
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
        pair_set = [self.node_to_pair(node) for node in path] # convert node identifier to pair of item weight and bin number
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
    args = argparse.ArgumentParser(description="ACO Bin Packing Problem")
    args.add_argument("-p", "--problem", type=int, default=1)
    args.add_argument("-a", "--ants", type=int, default=10)
    args.add_argument("-e", "--e_rate", type=float, default=0.9)
    parse = args.parse_args()
    problem = parse.problem
    p = parse.ants
    e = parse.e_rate
    if problem not in [1, 2] or p < 1 or e < 0 or e > 1:
        raise ValueError("Invalid parameters.")
    
    bpp = ACOBinPacker(problem, p, e)
    bpp.run()
    print(f"ACO Bin Packing Problem {problem}\n{'=' * 20}")
    print(f"Parameters: Ants={p}, Evaporation Rate={e}")
    print("Results:")
    print(f"Best Fitness: {bpp.best_fitness}")
    print(f"Worst Fitness: {bpp.worst_fitness}")
    print(f"Average Fitness: {bpp.avg_fitness}")


if __name__ == '__main__':
    main()
