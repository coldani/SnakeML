import time
from math import sqrt

import numpy as np

from main import Game
from ML_controller import FeedForwardNetwork


class GeneticAlgorithm:
    def __init__(
            self, pop_size: int, layers_size: list[int]) -> None:
        """Implements a genetic algorithm used to train a neural network on the
        Snake game. 
        Specifically, it implements a "(mu + lambda) - ES" model, with adaptive
        sigma (standard deviation of mutation).

        Args:
            -pop_size (int): Size of the population on each epoch
            -layers_size (list[int]): Number of neurons (exluding the bias) 
                on each network layer, excluding the output layer
        """

        self.pop_size: int = pop_size
        layers_size.append(3)  # final layer with 3 directions
        self.layers_size: list[int] = layers_size

        self.parents_size: int = pop_size // 6
        self.offspring_size: int = pop_size - self.parents_size
        self.tau: float = 1 / sqrt(
            FeedForwardNetwork.calc_num_weights(layers_size))
        self.population: np.ndarray = self.generate_random_population()
        self.best_fitness_evolution: list[int] = []
        self.avg_fitness_evolution: list[int] = []
        self.fittest_individual: list[np.ndarray] = []

    def generate_random_population(self) -> np.ndarray:
        """Helper function used to initialise a random population (i.e. a random
        matrix of weights and sigma)

        Returns:
            np.ndarray: Population matrix, where each row represents an individual,
            all but the last element of each row are the neural network weights,
            and the last element of each row is sigma (standard deviation of
            mutation)
        """
        weights: np.ndarray = FeedForwardNetwork.random_weights(
            self.pop_size, self.layers_size)
        sigma: np.ndarray = np.random.random((self.pop_size, 1))
        return np.append(weights, sigma, axis=1)

    def select_parents(self, fitness: np.ndarray) -> np.ndarray:
        """Returns the self.parents_size fittest individuals of the population

        Args:
            fitness (np.ndarray): Array of size self.pop_size with fitness of
            each individual

        Returns:
            np.ndarray: View of self.population with only the fittest individuals
        """
        indices = np.argpartition(
            fitness, self.pop_size - self.parents_size)[-self.parents_size:]
        return self.population[indices, :]

    def generate_offsprings(self, parents: np.ndarray) -> np.ndarray:
        """Creates a new generation of offsprings by selecting randomly from
        parents, updating sigma and applying mutation

        Args:
            parents (np.ndarray): Parents from which to select offsprings before 
            mutation

        Returns:
            np.ndarray: New matrix of offsprings with self.offspring_size
            number of rows
        """
        # Select offsprings randomly from parents
        indices = np.random.choice(
            self.parents_size, (self.offspring_size,),
            replace=True)
        offsprings = parents[indices, :].copy()

        # Update sigma
        mutation_of_sigma = np.random.normal(0, 1, self.offspring_size)
        sigma = offsprings[:, -1] * np.exp(self.tau * mutation_of_sigma)
        offsprings[:, -1] = sigma

        # Apply random mutation based on new sigma
        mutation = np.random.normal(
            0, sigma[:, np.newaxis],
            (offsprings.shape[0], offsprings.shape[1] - 1))
        offsprings[:, :-1] = offsprings[:, :-1] + mutation

        return offsprings

    def calculate_fitness(
            self, num_matches: int, stuck_threshold: int) -> np.ndarray:
        """Calculates fitness for each individual. Fitness is defined as the 
        sum of the score of num_matches runs of Snake.

        Args:
            num_matches (int): The number of matches to play to 
                calculate an individual's fitness
            stuck_threshold (int): The stuck_counter_threshold used in Game,
                a lower threshold results in quicker training as the game is
                stopped earlier in case of no progress.

        Returns:
            np.ndarray: Array with fitness for each individual
        """
        fitness = np.zeros(self.pop_size)
        for individual in range(self.pop_size):
            weights = self.population[individual, :-1][np.newaxis, :]
            for match in range(num_matches):
                # Note we don't want to reposition the apple after training
                game = Game(False, weights, self.layers_size, stuck_threshold+1)
                game.run(False, stuck_threshold)
                fitness[individual] += game.model.score

        return fitness

    def new_population(self, fitness: np.ndarray) -> np.ndarray:
        """Calculates next generation by applying the (mu+lambda)-ES algorithm

        Args:
            fitness (np.ndarray): Array of population fitness

        Returns:
            np.ndarray: Matrix representing new population, with self.pop_size
            rown
        """
        # select most fit as parents
        parents = self.select_parents(fitness)

        # generate offsprings with mutation from parents
        offsprings = self.generate_offsprings(parents)

        # return new population
        return np.append(parents, offsprings, axis=0)

    def train(
            self, epochs: int, num_matches: int = 1, stuck_threshold: int = 200,
            print_frequency: int = 0):
        """Trains the neural network applying the (mu+lambda)-ES algorithm
        On each epoch saves the fittest individual along with its fitness

        Args:
            -epochs (int): Number of training epochs
            -num_matches (int, optional): Number of Snake matches to play to 
                calculate an individual's fitness. Defaults to 1.
            -stuck_threshold (int, optional): The stuck_counter_threshold used
                in each game to calculate fitness, a lower threshold results in 
                quicker training as the game is stopped earlier in case of no 
                progress.
            -print_frequency (int, optional): If > 0, prints fitness of best 
            individual every print_frequency epoch. Defaults to 0 (i.e. nothing 
            is printed).
        """

        start = time.process_time()

        for epoch in range(epochs):
            fitness = self.calculate_fitness(num_matches, stuck_threshold)
            self.population = self.new_population(fitness)

            self.best_fitness_evolution.append(np.max(fitness))
            self.avg_fitness_evolution.append(np.average(fitness))
            fittest_individual = self.population[np.argmax(
                fitness), :][np.newaxis, :]
            self.fittest_individual.append(fittest_individual)

            if print_frequency > 0 and epoch % print_frequency == 0:
                print(f"Epoch: {epoch}" +
                      f" - max fitness: {np.max(fitness)}," +
                      f" avg fitness: {np.average(fitness)}")

        end = time.process_time()
        elapsed = end - start
        if print_frequency > 0:
            hour = elapsed // 3600
            min = (elapsed % 3600) // 60
            sec = (elapsed % 3600) % 60
            print(
                f"Trained {epochs} epochs " +
                f"(population size: {self.pop_size}, games played: {num_matches}) " +
                f"in {hour:.0f}h {min:.0f}m {sec:.0f}s")
