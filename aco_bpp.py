'''
BPP1: Bin Packing Problem 1
- 500 items
- weight of item i is i (e.g. item 1 has weight 1, item 2 has weight 2, etc.)
- 10 bins

BPP2: Bin Packing Problem 2
- 500 items
- weight of item i is (i^2)/2 (e.g. item 1 has weight 0.5, item 2 has weight 2, etc.)
- 50 bins

Instructions to run:
- Python version:
    Minimum: 3.6
    Recommended: 3.12
- Install required packages with the following command:
    pip install -r requirements.txt
- Run the script with the following command:
    python aco_bpp.py -p <problem_number> -a <ants> -e <evaporation_rate>
- problem_number: 1 or 2
- ants: number of ants to use in 1 trial
- evaporation_rate: between 0 and 1
- Example: python aco_bpp.py -p 1 -a 10 -e 0.9
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
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
        self.no_of_items = 500
        self.no_of_bins = 10 if problem == 1 else 50
        self.e_rate = e_rate
        self.ants = ants
        self.p_matrix = self.initialize_pheromones()
        self.best_fitness = np.inf
        self.worst_fitness = np.inf
        self.avg_fitness = np.inf
        self.fitness_values_graph = []

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
            current_run_fitness = []
            for j in runs:
                path, fitness = j
                current_run_fitness.append(fitness)
                self.pheromone_update(path, fitness)
            self.apply_evaporation()
            all_fitness.append(current_run_fitness)
        temp = [x for xs in all_fitness for x in xs]
        self.best_fitness = min(temp)
        self.worst_fitness = max(temp)
        self.avg_fitness = sum(temp) / len(temp)
        self.fitness_values_graph = all_fitness

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
        for _ in range(self.no_of_items):
            bin_choices = self.p_matrix[path[-1]] # refer to row of current node to find next bin choices
            total_pheromone = sum(bin_choices)
            normalized_pheromones = [pheromone / total_pheromone for pheromone in bin_choices]
            selected_node = random.choices(list(range(len(bin_choices))), normalized_pheromones)[0]
            path.append(selected_node)
        return path
     
    def fitness_eval(self, path: list):
        """evaluate fitness by getting difference between heaviest and lightest bins
        
        Parameters:
            path: list of node integers
        
        Returns:
            Value of diff between heaviest and lightest bin 
        """
        bins = {}
        pair_set = [self.node_to_pair(node) for node in path if node != 0] # convert node identifier to pair of item weight and bin number
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
        item = ((node - 1) // self.no_of_bins) + 1
        bin = node % self.no_of_bins
        return (item, bin)

    def plot_graph(self):
        """Plot graph to show fitness values over time for particular instance
        
        Parameters:
            fitness_values: list of lists of fitness values
        """
        if not self.fitness_values_graph:
            return
        iterations = list(range(1, len(self.fitness_values_graph) + 1))
        avg_fitness = [sum(values) / len(values) for values in self.fitness_values_graph]
        min_fitness = [min(values) for values in self.fitness_values_graph]
        max_fitness = [max(values) for values in self.fitness_values_graph]
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, avg_fitness, label="Average Fitness", color="blue", linewidth=2)
        plt.fill_between(iterations, min_fitness, max_fitness, color="lightblue", alpha=0.4, label="Fitness Range (Min to Max)")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Value")
        plt.title("Fitness Value Over Time")
        plt.legend()
        plt.savefig(f"bpp{self.problem}_fitness_plt_p{self.ants}_e{self.e_rate}.png", format="png", dpi=300)


def main():
    args = argparse.ArgumentParser(description="ACO Bin Packing Problem")
    args.add_argument("-p", "--problem", type=int, default=1)
    args.add_argument("-a", "--ants", type=int, default=10)
    args.add_argument("-e", "--e_rate", type=float, default=0.9)
    args.add_argument("-g", "--graph", action="store_true")
    parse = args.parse_args()
    problem = parse.problem
    p = parse.ants
    e = parse.e_rate
    if problem not in [1, 2] or p < 1 or e < 0 or e > 1:
        raise ValueError("Invalid parameters.")
    
    print(f"Running ACO Bin Packing Problem {problem} with {p} ants and evaporation rate of {e}")
    bpp = ACOBinPacker(problem, p, e)
    bpp.run()
    print("Results:")
    print(f"Best Fitness: {bpp.best_fitness}")
    print(f"Worst Fitness: {bpp.worst_fitness}")
    print(f"Average Fitness: {bpp.avg_fitness}")
    if parse.graph:
        bpp.plot_graph()


if __name__ == '__main__':
    main()
