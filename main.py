from typing import List

import numpy as np
import pygame

from keyboard_controller import Controller, KeyboardController
from ML_controller import MLController
from snake_model import SnakeModel
from snake_views import GameSurface, Window


class Game:
    FPS: int = 8

    def __init__(
            self, human: bool = True, weights: np.ndarray = None,
            layers_size: List[int] = None, apple_reposition_rate: int = 50,
            initial_snake_length: int = 1, snake_life_gain: int = 50):
        """Class that implements the snake game. Can be played either by a human
        with the keyboard or by the computer, with the direction updated by a 
        feedforward neural network

        Args:
            human (bool, optional): If True, snake is moved from the keyboard, 
                otherwise it will move on its own. Defaults to True.
            weights (np.ndarray, optional): Unrolled weights for the neural
                network, only used if human == False. 
                Defaults to None.
            layers_size (List[int], optional): Number of neurons for each layer
                of the neural network, only used if human == False . 
                Defaults to None.
            apple_reposition_rate (int, optional): Number of steps after which to
                reposition the apple. Defaults to 50.
            initial_snake_length (int, optional): initial snake length. 
                Defaults to 1.
            snake_life (int, optional): snake life gain when apple is eaten.
                Initial life is twice this value. Defaults to 50.
        """

        self.model: SnakeModel = SnakeModel(
            apple_reposition_rate,
            GameSurface.WIDTH,
            GameSurface.HEIGHT,
            initial_snake_length,
            snake_life_gain)
        if human:
            self.controller: Controller = KeyboardController(self.model)
        else:
            assert weights is not None, "Must provide weights"
            assert layers_size is not None, "Must provide layer size"
            self.controller: Controller = MLController(
                self.model, weights, layers_size)

    def run(self, display=True):
        """Runs the game

        Args:
            display (bool, optional): If True, it displays the game. Only set to
                False when training the neural network.
                Defaults to True.
        """
        if display:
            window: Window = Window()

        running: bool = True
        while running:
            if display:
                pygame.time.Clock().tick(Game.FPS)
                if pygame.event.peek(pygame.QUIT):
                    running = False
            self.controller.update_direction()
            if display:
                window.update(self.model)
            if (self.model.snake_dead
               or self.model.victory
               or self.model.snake_life < 0):
                running = False
        if display:
            pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
