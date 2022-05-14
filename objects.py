# TODO list:
# -togli possibilità di far nascere apple dove c'è snake (set difference)
# -create X (in SnakeML), add/create X to "calc_next_direction" method in SnakeML
# -create method in Game that makes the computer play and display it
# -parallelise training
# -documentation!!
# -add some visualisation tools (score / fitness after each epoch, etc)

import pygame
import random
import itertools
import numpy as np
import genetic_algorithm as ga
from constants import *


class Window:
    def __init__(self, training=False):
        self.width = WIDTH * GRID_SIZE
        self.height = HEIGHT * GRID_SIZE
        self.grid_size = GRID_SIZE
        self.background_color = BACKGROUND_COLOR
        self.caption = CAPTION
        self.training = training

        if not training:
            self.window = pygame.display.set_mode((self.width, self.height))
            self.window.fill(self.background_color)
            pygame.display.set_caption(self.caption)

    def _update(self):
        if not self.training:
            self.window.fill(self.background_color)


class Snake:
    def __init__(self, training=False):
        self.training = training
        self.window = Window(training)
        self.color = SNK_COLOR
        self.head_color = SNK_HEAD_COLOR
        self.hedge_length = GRID_SIZE
        self.speed = SNK_SPEED
        self.positions = [SNK_INIT_POS]
        self.directions = [SNK_INIT_DIR]
        self.score = 0

        self.update_rate = self.hedge_length // self.speed
        self.cycle_count = 0

        self.snake_len = 1
        self.growth_cycle_handler = {}
        self.growth_flag = False
        self.new_positions = {}
        self.new_directions = {}

    def _update(self, new_direction):

        # adjust directions
        _ = self.directions.pop(-1)
        self.directions.insert(0, new_direction)

        # move the snake
        for i, position in enumerate(self.positions):
            direction = self.directions[i * self.update_rate]
            x, y = position
            x += direction[0] * self.speed
            y += direction[1] * self.speed
            position = (x, y)
            self.positions[i] = position

        # grow the snake
        if self.growth_flag:
            if next(self.growth_cycle_handler[self.snake_len + 1]) == self.update_rate - 1:
                self.positions.append(self.new_positions[self.snake_len + 1])

                del self.growth_cycle_handler[self.snake_len + 1]
                del self.new_positions[self.snake_len + 1]
                self.snake_len += 1

                if len(self.growth_cycle_handler) == 0:
                    self.growth_flag = False

        # redraw the snake
        if not self.training:
            for i, position in enumerate(self.positions):
                if i == 0:
                    color = self.head_color
                else:
                    color = self.color
                body_piece = pygame.Rect(position, (self.hedge_length, self.hedge_length))
                pygame.draw.rect(self.window.window, color, body_piece, 0)
                pygame.draw.rect(self.window.window, (0, 0, 0), body_piece, 1)

    def check_eat(self, apple):
        apl_x, apl_y = apple.center
        apl_radius = apple.radius
        head_position = self.positions[0]

        # yumm!
        if (
            (head_position[0] < apl_x + apl_radius)
            and (head_position[0] + self.hedge_length > apl_x - apl_radius)
            and (head_position[1] < apl_y + apl_radius)
            and (head_position[1] + self.hedge_length > apl_y - apl_radius)
        ):
            self.score += 1
            self.grow()
            return True
        else:
            return False

    def check_hit(self):
        head_position = self.positions[0]
        hit = False

        # hit the border
        if (
            (head_position[0] < 0)
            or (head_position[0] + self.hedge_length > self.window.width)
            or (head_position[1] < 0)
            or (head_position[1] + self.hedge_length > self.window.height)
        ):
            hit = True

        # hit the body
        if self.snake_len > 2:
            for position in self.positions[2:]:
                if (
                    (head_position[0] < position[0] + self.hedge_length)
                    and (head_position[0] + self.hedge_length > position[0])
                    and (head_position[1] < position[1] + self.hedge_length)
                    and (head_position[1] + self.hedge_length > position[1])
                ):
                    hit = True
        return hit

    def grow(self):
        self.growth_cycle_handler[self.snake_len + 1] = iter(range(self.update_rate))
        self.new_positions[self.snake_len + 1] = self.positions[-1]
        self.directions.extend([None for x in range(self.update_rate)])
        self.growth_flag = True


class SnakeML(Snake):
    num_labels = 3
    input_nodes = None  # TODO

    def __init__(self, weights, training=True, add_bias=True):
        Snake.__init__(training)

        self.weights = weights
        self.num_layers = len(weights) + 1
        # self.nodes_hidden_layers = [weight.shape[1] for key, weight in enumerate(weights)]
        # self.nodes_layers = np.append(self.input_nodes, self.nodes_hidden_layers)
        # self.nodes_layers = np.append(self.nodes_layers, self.num_labels)
        # self.nodes_layers = list(map(int, self.nodes_layers))
        self.add_bias = add_bias
        self.countdown = 200
        self.stop_training = False
        self.a = {}
        self.z = {}

    @classmethod
    def reshape_weights(cls, weights_array, size_inner_layers, add_bias):
        size_layers = size_inner_layers
        size_layers.insert(0, cls.input_nodes)
        size_layers.append(cls.num_labels)
        weights = {}
        for layer in range(1, size_layers):
            size = (size_layers[layer - 1] + add_bias) * size_layers[layer]
            weights[layer] = weights_array[:size].reshape(
                (size_layers[layer], size_layers[layer - 1] + add_bias)
            )
            weights_array = weights_array[size:]

        return weights

    def calc_next_direction(self):
        X = None  # TODO

        y_hat = self._forward_propagation(X)
        nn_choice = np.argmax(y_hat)
        current_direction = self.directions[0]

        if current_direction == SNK_DIR["UP"]:
            if nn_choice == 0:
                new_direction = SNK_DIR["LEFT"]
            elif nn_choice == 1:
                new_direction = current_direction
            elif nn_choice == 2:
                new_direction = SNK_DIR["RIGHT"]
        elif current_direction == SNK_DIR["RIGHT"]:
            if nn_choice == 0:
                new_direction = SNK_DIR["UP"]
            elif nn_choice == 1:
                new_direction = current_direction
            elif nn_choice == 2:
                new_direction = SNK_DIR["DOWN"]
        elif current_direction == SNK_DIR["DOWN"]:
            if nn_choice == 0:
                new_direction = SNK_DIR["RIGHT"]
            elif nn_choice == 1:
                new_direction = current_direction
            elif nn_choice == 2:
                new_direction = SNK_DIR["LEFT"]
        elif current_direction == SNK_DIR["LEFT"]:
            if nn_choice == 0:
                new_direction = SNK_DIR["DOWN"]
            elif nn_choice == 1:
                new_direction = current_direction
            elif nn_choice == 2:
                new_direction = SNK_DIR["UP"]

        return new_direction

    def _update(self, new_direction):
        Snake._update(self, new_direction)
        self.countdown <= 1

    def check_eat(self, apple):
        yum = Snake.check_eat(self, apple)
        if yum:
            self.countdown = 200
        return yum

    def _ReLU(self, z):
        """
        z: np ndarray
        returns: rectified linear function of z
        """
        relu_ = np.maximum(0.0, z)
        return relu_

    def _softmax(self, z):
        """
        z: np ndarray, with observations in the rows, labels in the columns
        returns: softmax function of z 
        """
        z -= np.max(z, axis=1, keepdims=True)  # trick for numerical stability
        softmax = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return softmax

    def _forward_propagation(self, X):
        """
        Calculates all a[layer] vectors, where a[0] is the input layer and a[num_layers] is the fitted output layer
        """
        weights = self.weights
        z = {}
        a = {}
        a[0] = X
        for layer in range(self.num_layers - 1):
            if self.add_bias:
                bias = np.ones((X.shape[0], 1))
                z[layer + 1] = np.append(bias, a[layer], axis=1)
            else:
                z[layer + 1] = a[layer]
            z[layer + 1] = np.matmul(z[layer + 1], weights[layer + 1].T)

            if layer == self.num_layers - 2:
                funct = self._softmax
            else:
                funct = self._ReLU
            a[layer + 1] = funct(z[layer + 1])

        y_hat = a[self.num_layers - 1]

        return y_hat


class Apple:
    def __init__(self, training=False):
        self.training = training
        self.window = Window(training)
        self.color = APL_COLOR
        self.radius = APL_RADIUS
        self.center = self.recenter()
        self.recenter_count = 0

    def _update(self, recenter):
        if recenter:
            self.center = self.recenter()
        if not self.training:
            pygame.draw.circle(self.window.window, self.color, self.center, self.radius, 0)
            pygame.draw.circle(self.window.window, (0, 0, 0), self.center, self.radius, 1)

    def recenter(self):
        grid_size = self.window.grid_size
        grid_w = self.window.width // grid_size - 1
        grid_h = self.window.height // grid_size - 1
        # x = random.randint(0, grid_w) * grid_size + self.radius
        # y = random.randint(0, grid_h) * grid_size + self.radius
        # return (x, y)
        grid = list(itertools.product(range(grid_w), range(grid_h)))
        i = np.random.choice(len(grid))
        x, y = grid[i]
        x = x * grid_size + self.radius
        y = y * grid_size + self.radius

        return (x, y)


class Game:
    def __init__(self, training=False):
        self.window = Window(training)
        self.apple = Apple(training)
        self.training = training
        self.running = True

    def _update(self, apl_recenter, snk_direction):
        self.window._update()
        self.apple._update(apl_recenter)
        self.snake._update(snk_direction)
        if self.snake.check_eat(self.apple):
            self.apple._update(True)
            self.apple.recenter_count = 0
        if not self.training:
            pygame.display.flip()

    def run(self):
        self.snake = Snake(False)
        self.grid_cycle = itertools.cycle(range(self.window.grid_size // self.snake.speed))
        running = True
        key = None
        snk_direction = self.snake.directions[0]

        while running:
            pygame.time.Clock().tick(FPS)

            apl_recenter = False
            self.apple.recenter_count += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    key = event.key

            new_direction = snk_direction
            count_cycle = next(self.grid_cycle)
            if count_cycle == 0:
                if (key == pygame.K_LEFT) and not (snk_direction == SNK_DIR["RIGHT"]):
                    new_direction = SNK_DIR["LEFT"]
                if (key == pygame.K_RIGHT) and not (snk_direction == SNK_DIR["LEFT"]):
                    new_direction = SNK_DIR["RIGHT"]
                if (key == pygame.K_UP) and not (snk_direction == SNK_DIR["DOWN"]):
                    new_direction = SNK_DIR["UP"]
                if (key == pygame.K_DOWN) and not (snk_direction == SNK_DIR["UP"]):
                    new_direction = SNK_DIR["DOWN"]
            snk_direction = new_direction

            if self.apple.recenter_count >= (APL_UPDATE_RATE * FPS):
                apl_recenter = True
                self.apple.recenter_count = 0

            self._update(apl_recenter, snk_direction)

            if self.snake.check_hit():
                running = False

    def auto_run(self, snakeML):
        self.grid_cycle = itertools.cycle(range(self.window.grid_size // snakeML.speed))
        running = True
        snk_direction = snakeML.directions[0]

        while running:
            new_direction = snk_direction
            count_cycle = next(self.grid_cycle)
            if count_cycle == 0:
                new_direction = snakeML.calc_next_direction()
            snk_direction = new_direction
            self.snake._update(snk_direction)
            if self.snake.check_eat(self.apple):
                self.apple._update(True)
            if snakeML.check_hit() or snakeML.countdown <= 0:
                running = False

        return snakeML.score

    def _calculate_fitness(self, population, size_inner_layers, add_bias):
        fitness = np.empty(population.shape[0])
        for brain in range(population.shape[0]):
            weights = SnakeML.reshape_weights(brain, size_inner_layers, add_bias)
            snake = SnakeML(weights, True, add_bias)
            score = self.auto_run(snake)
            fitness[brain] = score

        return fitness

    def train(
        self,
        pop_size=1000,
        max_epochs=1000,
        mut_prob=0.3,
        cross_prob=0.3,
        surv_pct=0.1,
        surv_non_mut_pct=0.1,
        parents_pct=0.4,
        add_bias=True,
        size_inner_layers=[18, 18],
    ):
        input_nodes = SnakeML.input_nodes
        output_nodes = SnakeML.output_nodes
        layers_nodes = [input_nodes] + size_inner_layers + output_nodes
        num_weights = 0
        for i in range(layers_nodes - 1):
            num_weights += (layers_nodes[i] + int(add_bias)) * layers_nodes[i + 1]

        initial_pop = np.random.random(pop_size, num_weights)
        fitness_function = self._calculate_fitness
        fitness_params = (initial_pop, size_inner_layers, add_bias)
        training = ga.genetic_algorithm(initial_pop, fitness_function, *fitness_params)
        training.train(
            parents_pct,
            surv_pct,
            surv_non_mut_pct,
            cross_prob=cross_prob,
            mut_prob=mut_prob,
            max_epochs=max_epochs,
            verbose=1,
        )
        best_w = training.select_best(
            training.population, 1, training.fitness
        )  # this will not be needed actually as I will have the array of best for each epoch
        # so just select the best for the last epoch and this is the chosen snake

