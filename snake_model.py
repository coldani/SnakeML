import enum
import itertools
import random
from typing import Tuple

Position = Tuple[int, int]  # [left, top]


class Directions(enum.Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)


class SnakeModel:
    def __init__(self, apple_reposition_rate: int, width: int, height: int):
        """
        The model that recalculates object positions and game status after each
        step.

        Args:
            apple_reposition_rate (int): number of steps after which apple is
            repositioned
            width (int, optional): Width of the grid
            height (int, optional): height of the grid
        """
        self.width: int = width
        self.height: int = height
        self.apple_reposition_rate: int = apple_reposition_rate

        self.grid: set[Position] = set(
            itertools.product(range(width), range(height)))
        self.snake_positions: list[Position] = [(width // 2, height // 2)]
        self.apple_position: Position = self.random_free_position()

        self.snake_dead: bool = False
        self.victory: bool = False
        self.score: int = 0
        self.apple_reposition_counter: int = 0

    def free_positions(self) -> set[Position]:
        """
        Returns the set of positions not occupied by the snake.
        Note it includes the position of the apple

        Returns:
            set[Position]: set of positions not occupied by the snake
        """
        return self.grid.difference(set(self.snake_positions))

    def step(self, direction: Directions):
        """
        Moves the snake by one position, based on the input direction

        Args:
            direction (Directions): direction in which to move the snake
        """
        if self.victory or self.snake_dead:
            return

        grow_flag = False
        new_position = self.next_position(self.snake_positions[0], direction)
        if new_position not in self.free_positions():
            self.snake_dead = True
        elif self.is_apple_eaten(new_position):
            self.score += 1
            grow_flag = True
            self.reposition_apple()
        elif self.apple_reposition_counter > self.apple_reposition_rate:
            self.reposition_apple()

        self.move_snake(new_position, grow_flag)
        if len(self.free_positions()) == 0:
            self.victory = True
        self.apple_reposition_counter += 1

    def move_snake(self, new_position: Position, grow_flag: bool):
        """Recomputes snake positions

        Args:
            new_position (Position): new position for snake head
            grow_flag (bool): if true, it means that the last position of the
            snake (the tail) is not removed
        """
        self.snake_positions.insert(0, new_position)
        if not grow_flag:
            self.snake_positions.pop(-1)

    def reposition_apple(self):
        """
        Changes apple_position to a new random free position
        """
        self.apple_reposition_counter = 0
        self.apple_position = self.random_free_position()

    def random_free_position(self) -> Position:
        """Return a random free position

        Returns:
            Position: A random free position
        """
        return random.choice(tuple(self.free_positions()))

    def is_apple_eaten(self, new_position: Position) -> bool:
        """Checks whether the snake eats the apple

        Args:
            new_position (Position): New snake head position after movement

        Returns:
            bool: True if snake eats the apple, false otherwise
        """
        return self.apple_position == new_position

    def next_position(
            self, position: Position, direction: Directions) -> Position:
        """
        Calculates next position based on direction

        Args:
            position (Position): Starting position
            direction (Directions): Moving

        Returns:
            Position: New position
        """
        return (position[0]+direction.value[0], position[1]+direction.value[1])
