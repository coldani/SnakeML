import string
from typing import Tuple

import pygame

from snake_model import Position, SnakeModel

Color = Tuple[int, int, int]


class Window:

    def __init__(self):
        LEFT_PADDING: int = 1
        RIGHT_PADDING: int = 1
        TOP_PADDING: int = 3
        BOTTOM_PADDING: int = 1
        CAPTION: string = "Snake"
        self.BACKGROUND_COLOR: Color = (200, 200, 200)

        pygame.init()
        self.width: int = (
            GameSurface.WIDTH + LEFT_PADDING + RIGHT_PADDING) * \
            GameSurface.GRID_SIZE
        self.height: int = (
            GameSurface.HEIGHT + TOP_PADDING + BOTTOM_PADDING) * \
            GameSurface.GRID_SIZE
        self.surface: pygame.Surface = pygame.display.set_mode(
            (self.width, self.height))

        self.left_offset: int = LEFT_PADDING * GameSurface.GRID_SIZE
        self.top_offset: int = TOP_PADDING * GameSurface.GRID_SIZE
        self.game_surface: GameSurface = GameSurface(
            self)
        self.snake: Snake = Snake(self)
        self.apple: Apple = Apple(self)

        self.surface.fill(self.BACKGROUND_COLOR)
        pygame.display.set_caption(CAPTION)

    def update(self, model: SnakeModel):
        self.surface.fill(self.BACKGROUND_COLOR)
        self.game_surface.update()
        self.snake.update(model.snake_positions)
        self.apple.update(model.apple_position)
        pygame.display.flip()

    def scale_position(self, position: Position) -> Position:
        """Utility function that rescales object position

        Args:
            position (Position): position in the grid

        Returns:
            Position: position rescaled based on size of grid
        """
        x, y = position
        x = x * GameSurface.GRID_SIZE + self.left_offset
        y = y * GameSurface.GRID_SIZE + self.top_offset
        return (x, y)


class GameSurface:
    WIDTH, HEIGHT = (30, 30)
    GRID_SIZE: int = 15

    def __init__(self, window: Window):
        """Main view that is responsible for drawing all objects"""
        self.BACKGROUND_COLOR: Color = (240, 240, 240)

        self.window: Window = window

        self.width: int = GameSurface.WIDTH * GameSurface.GRID_SIZE
        self.height: int = GameSurface.HEIGHT * GameSurface.GRID_SIZE

        self.surface: pygame.Rect = pygame.Rect(
            self.window.scale_position((0, 0)), (self.width, self.height))

        pygame.draw.rect(
            self.window.surface, self.BACKGROUND_COLOR, self.surface, 0)
        pygame.draw.rect(self.window.surface, (0, 0, 0), self.surface, 1)

    def update(self):
        """Update the views by redrawing all objects

        Args:
            model (SnakeModel): SnakeModel from which relevant information on
            game status and object positions are taken
        """
        pygame.draw.rect(
            self.window.surface, self.BACKGROUND_COLOR, self.surface, 0)
        pygame.draw.rect(self.window.surface, (0, 0, 0), self.surface, 1)


class Snake:

    def __init__(self, window: Window):
        self.HEAD_COLOR: Color = (220, 220, 20)
        self.COLOR: Color = (100, 220, 100)

        self.window: Window = window

    def is_inside_surface(self, position: Position) -> bool:
        x = position[0]
        y = position[1]
        return (x >= 0 and y >= 0 and
                x < GameSurface.WIDTH and y < GameSurface.HEIGHT)

    def update(self, positions: list[Position]):
        for i, position in enumerate(positions):
            if self.is_inside_surface(position):
                position = self.window.scale_position(position)
                color = self.HEAD_COLOR if i == 0 else self.COLOR
                body_piece = pygame.Rect(
                    position, (GameSurface.GRID_SIZE, GameSurface.GRID_SIZE))
                pygame.draw.rect(self.window.surface, color, body_piece, 0)
                pygame.draw.rect(self.window.surface, (0, 0, 0), body_piece, 1)


class Apple:

    def __init__(self, window: Window):
        self.COLOR: Color = (255, 50, 50)
        self.RADIUS: float = GameSurface.GRID_SIZE / 2

        self.window: Window = window

    def adjust_centre(self, centre: Position) -> Position:
        x = centre[0]
        y = centre[1]
        return (x+self.RADIUS, y+self.RADIUS)

    def update(self, centre: Position):
        centre = self.window.scale_position(centre)
        centre = self.adjust_centre(centre)
        # draws circle
        pygame.draw.circle(self.window.surface, self.COLOR,
                           centre, self.RADIUS, 0)
        # draws edge
        pygame.draw.circle(
            self.window.surface, (0, 0, 0),
            centre, self.RADIUS, 1)
