import numpy as np

from keyboard_controller import Controller
from snake_model import Directions, SnakeModel


class MLController(Controller):
    def __init__(self, model: SnakeModel, weights: np.ndarray,
                 layers_size: list[int]):
        """Implements ML controller

        Args:
            model (SnakeModel): the instance of the model used for the game
            weights (np.ndarray): Unrolled weights for the neural network
            layers_size (list[int]): List with number of neurons for each layer 
                                    of the network. Output layer size must be 
                                    equal to 3.
        """
        self.model: SnakeModel = model
        self.weights: np.ndarray = weights
        self.layers_size: list[int] = layers_size

        self.direction: Directions = Directions.RIGHT
        assert self.layers_size[-1] == 3, "Output layer must have 3 neurons"
        assert self.layers_size[0] == self.get_inputs(
        ).shape[1], "Check input size"

    def get_inputs(self) -> np.ndarray:
        """Function that generates the input layer for the neural network

        Returns:
            np.ndarray: Input layer, with shape [1, m]
        """
        m = self.model
        dir = self.direction

        dist_to_apple = m.normalised_relative_distance_to(dir, m.apple_position)
        dist_to_obstacles = m.normalised_distance_to_obstacles(dir)
        snake_length = m.normalise_value(
            m.snake_length(), 1, m.width * m.height, -1, 1)
        regions_density = m.normalised_regions_density(dir, 4)

        input = np.array([*dist_to_apple,
                          *dist_to_obstacles,
                          snake_length,
                          *regions_density])[np.newaxis, :]
        return input

    def update_direction(self):
        """Updates snake direction based on output from neural network"""

        out = FeedForwardNetwork.feed_forward(
            self.weights, self.get_inputs(), self.layers_size)
        if np.argmax(out) == 0:
            pass  # no changes to self.direction
        elif np.argmax(out) == 1:
            self.direction = self.direction.right()
        else:  # np.argmax(out) == 2
            self.direction = self.direction.left()

        self.model.step(self.direction)


class FeedForwardNetwork:
    """Class with only static methods that implements a simple feedforward 
    neural network
    """
    @staticmethod
    def num_neurons(layers_size: list[int], layer: int) -> tuple[int, int]:
        """Returns number of input and output neurons for a given layer

        Args:
            layers_size (list[int]): List with number of neurons for each layer 
                                    of the network
            layer (int): Index of input layer

        Returns:
            tuple[int, int]: Number of neurons the given layer and the next layer
        """
        return (layers_size[layer], layers_size[layer+1])

    @staticmethod
    def calc_num_weights(layers_size: list[int]) -> int:
        """Returns the total number of weights for a given network, including
        the bias

        Args:
            layers_size (list[int]): List with number of neurons for each layer
            of the network, excluding the bias

        Returns:
            int: Number of total weights for the network
        """
        num_weights: int = 0
        for i in range(len(layers_size)-1):
            n, next_n = FeedForwardNetwork.num_neurons(layers_size, i)
            num_weights += (n + 1) * next_n  # include +1 for bias
        return num_weights

    @staticmethod
    def random_weights(pop_size: int, layers_size: list[int]) -> np.ndarray:
        """Applies Xavier Glorot random weights initialisation. Note that given 
        this network is used for a genetic algorithm, weights are not shared 
        by individuals.

        Args:
            pop_size (int): Number of individuals (e.g. number of rows in the 
            weights matrix)
            layers_size (list[int]): List with number of neurons for each layer
            of the network, excluding the bias

        Returns:
            np.ndarray: Matrix with weights for all individuals. Each row 
            represents all the weights in the network for an individual, 
            including the bias
        """
        num_weights: int = FeedForwardNetwork.calc_num_weights(layers_size)
        weights_matrix: np.ndarray = np.empty((pop_size, num_weights))

        start: int = 0
        for i in range(len(layers_size) - 1):
            n, next_n = FeedForwardNetwork.num_neurons(layers_size, i)
            num_weights_layer: int = (n + 1) * next_n  # +1 for bias

            high: float = np.sqrt(6) / (n + next_n)
            low: float = -high
            weights_matrix[:, start: start + num_weights_layer] = np.random.uniform(
                low, high, (pop_size, num_weights_layer))
            start += num_weights_layer

        return weights_matrix

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Relu activation function"""
        return np.maximum(0.0, x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function

        Args:
            x (np.ndarray): each row is a different individual"""

        x -= np.max(x, axis=1, keepdims=True)  # trick for numerical stability
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def feed_forward(weights: np.ndarray, input: np.ndarray,
                     layers_size: list[int]) -> np.ndarray:
        """Computes one forward pass on the network. Activation function is relu
        for inner layers and softmax for output layer

        Args:
            weights (np.ndarray): One unrolled array of weights for the entire 
            network
            input (np.ndarray): Input layer
            layers_size (list[int]): List with number of neurons for each layer

        Returns:
            np.ndarray: Softmax activation of the output layer
        """
        assert input.shape[1] == layers_size[0]

        x: np.ndarray = input
        start: int = 0
        for i in range(len(layers_size)-1):
            n, next_n = FeedForwardNetwork.num_neurons(layers_size, i)
            num_weights_layer = (n + 1) * next_n  # +1 for bias

            # create weights matrix (n+1 for bias)
            w = weights[:, start: start + num_weights_layer].reshape(
                (n+1, next_n))
            start += num_weights_layer

            # add bias to input layer
            bias = np.ones((x.shape[0], 1))
            x = np.append(x, bias, axis=1)

            # multiply input layer with weights
            x = np.matmul(x, w)

            # apply activation function
            if i < len(layers_size)-2:
                x = FeedForwardNetwork.relu(x)
            else:
                x = FeedForwardNetwork.softmax(x)

        return x
