import pygame

from keyboard_controller import Controller, KeyboardController
from snake_model import SnakeModel
from snake_views import GameSurface, Window


class Game:
    APPLE_REPOSITION_RATE: int = 50
    FPS: int = 8

    def __init__(self, human: bool = True):
        self.human: bool = human
        if human:
            self.window: Window = Window()
            self.model: SnakeModel = SnakeModel(
                Game.APPLE_REPOSITION_RATE,
                GameSurface.WIDTH,
                GameSurface.HEIGHT)
            self.controller: Controller = KeyboardController(self.model)
        else:
            # TODO add ML model
            pass

    def run(self):
        running: bool = True
        while running:
            if self.human:
                pygame.time.Clock().tick(Game.FPS)
            self.controller.update_direction()
            self.window.update(self.model)
            if (self.controller.quit
               or self.model.snake_dead
               or self.model.victory):
                running = False


game = Game(True)
game.run()
