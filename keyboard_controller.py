from abc import ABC, abstractmethod

import pygame

from snake_model import Directions, SnakeModel


class Controller(ABC):
    @abstractmethod
    def __init__(self, model: SnakeModel):
        pass

    @abstractmethod
    def update_direction(self):
        pass


class KeyboardController(Controller):
    def __init__(self, model: SnakeModel):
        """Implements keyboard controller

        Args:
            model (SnakeModel): the instance of the model used for the game
        """
        self.model: SnakeModel = model
        self.direction: Directions = Directions.RIGHT
        self.quit: bool = False

    def update_direction(self):
        """
        Updates the snake direction by listening to the keyboard and calls
        model.step(direction)
        """
        key: int = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            if event.type == pygame.KEYDOWN:
                key = event.key

        if (key == pygame.K_LEFT) and not (self.direction == Directions.RIGHT):
            self.direction = Directions.LEFT
        if (key == pygame.K_RIGHT) and not (self.direction == Directions.LEFT):
            self.direction = Directions.RIGHT
        if (key == pygame.K_UP) and not (self.direction == Directions.DOWN):
            self.direction = Directions.UP
        if (key == pygame.K_DOWN) and not (self.direction == Directions.UP):
            self.direction = Directions.DOWN

        self.model.step(self.direction)
