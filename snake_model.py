import enum
import itertools
import random
from typing import Tuple, List, Set

Position = Tuple[int, int]  # [left, top]


class Directions(enum.Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

    def right(self):
        if self == Directions.UP:
            return Directions.RIGHT
        elif self == Directions.RIGHT:
            return Directions.DOWN
        elif self == Directions.DOWN:
            return Directions.LEFT
        elif self == Directions.LEFT:
            return Directions.UP

    def left(self):
        if self == Directions.UP:
            return Directions.LEFT
        elif self == Directions.LEFT:
            return Directions.DOWN
        elif self == Directions.DOWN:
            return Directions.RIGHT
        elif self == Directions.RIGHT:
            return Directions.UP


class SnakeModel:
    def __init__(
            self, apple_reposition_rate: int, width: int, height: int,
            initial_length: int = 1):
        """
        The model that recalculates object positions and game status after each
        step.

        Args:
            apple_reposition_rate (int): number of steps after which apple is
                repositioned
            width (int): width of the grid
            height (int): height of the grid
            initial_length (int, optional): initial length of the snake. 
                Defaults to 1.

        """
        self.width: int = width
        self.height: int = height
        self.apple_reposition_rate: int = apple_reposition_rate

        self.grid: Set[Position] = set(
            itertools.product(range(width), range(height)))
        self.snake_positions: List[Position] = self.initialise_snake(
            initial_length)
        self.apple_position: Position = self.random_free_position()

        self.snake_dead: bool = False
        self.victory: bool = False
        self.score: int = 0
        self.apple_reposition_counter: int = 0
        self.snake_stuck_counter: int = 0
        self.steps_counter = 0

    def initialise_snake(self, length: int) -> List[Position]:
        """Initialises the snake with given length. Length is capped at `self.width`

        Args:
            length (int): Initial snake length

        Returns:
            List[Position]: List of snake positions
        """
        if length <= 0:
            length = 1
        elif length > self.width:
            length = self.width
        head_position = max(length-1, self.width//2)
        positions = []
        for pos in range(head_position, head_position-length, -1):
            positions.append((pos, self.height // 2))

        return positions

    def snake_length(self):
        """Returns the length of the snake"""
        return len(self.snake_positions)

    def free_positions(self) -> Set[Position]:
        """
        Returns the set of positions not occupied by the snake.
        Note it includes the position of the apple

        Returns:
            Set[Position]: set of positions not occupied by the snake
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
        # dead
        if new_position not in self.free_positions():
            self.snake_dead = True
        # eats apple
        elif self.is_apple_eaten(new_position):
            self.score += 1
            self.snake_stuck_counter = 0
            grow_flag = True
            self.reposition_apple()
        # apple moves
        elif self.apple_reposition_counter > self.apple_reposition_rate:
            self.reposition_apple()

        # make actual move, grow snake if necessary
        self.move_snake(new_position, grow_flag)

        # check if victory
        if len(self.free_positions()) == 0:
            self.victory = True

        # increase counters
        self.apple_reposition_counter += 1
        self.snake_stuck_counter += 1
        self.steps_counter += 1

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

    def normalise_value(self, value, min, max, a, b):
        """Returns the normalised value, where the smallest possible value is
        mapped to a and the largest to b

        Args:
            value: the input value to be normalised
            min: the minimum possible value
            max: the maximum possible value
            a: start of range
            b: end of range
        """
        return a + (value - min)*(b-a)/(max-min)

    def relative_distance_to(self, current_direction: Directions,
                             object_position: Position) -> Tuple[int, int]:
        """Returns the distance to an object, relative to the direction taken by
        the head of the snake. An object 2 positions ahead of the head of the
        snake would return (2, 0), an object 2 positions to the right would
        return (0, 2), an object 1 position back and to the left would return
        (-1, -1)

        Args:
            current_direction (Directions): direction of the head of the snake
            object_position (Position): position of the object on the grid

        Returns:
            Tuple[int, int]: (distance_ahead, distance_right) - negative values
            mean the object is behind and/or to the left
        """
        head_position: Position = self.snake_positions[0]
        if current_direction == Directions.DOWN:
            distance_ahead = object_position[1] - head_position[1]
            distance_right = head_position[0] - object_position[0]
        if current_direction == Directions.UP:
            distance_ahead = head_position[1] - object_position[1]
            distance_right = object_position[0] - head_position[0]
        if current_direction == Directions.RIGHT:
            distance_ahead = object_position[0] - head_position[0]
            distance_right = object_position[1] - head_position[1]
        if current_direction == Directions.LEFT:
            distance_ahead = head_position[0] - object_position[0]
            distance_right = head_position[1] - object_position[1]
        return (distance_ahead, distance_right)

    def normalised_relative_distance_to(self, current_direction: Directions,
                                        object_position: Position) -> Tuple[int, int]:
        """Returns the distance to an object normalised in range [-1, 1]"""
        distance_ahead, distance_right = self.relative_distance_to(
            current_direction,
            object_position)
        maxi = max(self.width, self.height)
        min = -maxi
        a = -1
        b = 1
        return (self.normalise_value(distance_ahead, min, maxi, a, b),
                self.normalise_value(distance_right, min, maxi, a, b))

    def distance_to_obstacles(self, current_direction: Directions) -> Tuple[int, int, int]:
        """Computes the distance to the closest obstacle from head of snake in
         each direction (ahead, right, left). Note minimum distance is 1 to make
         it consistent with `self.relative_distance_to()`

        Args:
            current_direction (Directions): Current direction of head of snake

        Returns:
            Tuple[int, int, int]: distance to closest obstacle in each direction
            (ahead, right, left)
        """
        dir_right: Directions = current_direction.right()
        dir_left: Directions = current_direction.left()

        dist_ahead: int = self.closest_obstacle(current_direction)
        dist_right: int = self.closest_obstacle(dir_right)
        dist_left: int = self.closest_obstacle(dir_left)

        return (dist_ahead, dist_right, dist_left)

    def normalised_distance_to_obstacles(
            self, current_direction: Directions) -> Tuple[float, float, float]:
        """Computes the distance to closes obstacle in each direction (ahead, 
        right, left) normalised in [-1, 1] range."""
        dist_ahead, dist_right, dist_left = self.distance_to_obstacles(
            current_direction)
        min = 1
        maxi = max(self.width, self.height)
        a = -1
        b = 1
        return (self.normalise_value(dist_ahead, min, maxi, a, b),
                self.normalise_value(dist_right, min, maxi, a, b),
                self.normalise_value(dist_left, min, maxi, a, b))

    def closest_obstacle(self, direction: Directions) -> int:
        """Returns the distance to the closest obstacle from head of snake in
        the given direction. Minimum distance is 1"""
        free_positions: Set[Position] = self.free_positions()
        position: Position = self.snake_positions[0]
        position = (position[0] + direction.value[0],
                    position[1] + direction.value[1])
        distance: int = 1
        while position in free_positions:
            distance += 1
            position = (position[0] + direction.value[0],
                        position[1] + direction.value[1])
        return distance

    def normalised_closest_obstacle(self, direction: Directions) -> int:
        """Returns the distance to the closest obstacle in the given direction,
        normalised in range [-1, 1]"""
        distance = self.closest_obstacle(direction)
        min = 1
        maxi = max(self.width, self.height)
        a = -1
        b = 1
        return self.normalise_value(distance, min, maxi, a, b)

    def regions_density(self, current_direction: Directions,
                        edge_length: int = 4) -> Tuple[int, int, int, int]:
        """Computes the number of free positions in the four regions ahead/right,
        backward/right, backward/left, ahead/left relative to head of snake

        Args:
            current_direction (Directions): Current direction of head of snake
            edge_length (int, optional): Length of the region edge, e.g. the
            regions are squares area equal to `edge_length*edge_length`.
            Defaults to 4.

        Returns:
            Tuple[int, int, int, int]: Number of free positions in regions
            (ahead/right, backward/right, backward/left, ahead/left) relative to
            head of snake
        """
        ahead_right = self.single_region_density(current_direction, edge_length)
        back_right = self.single_region_density(
            current_direction.right(), edge_length)
        back_left = self.single_region_density(
            current_direction.right().right(), edge_length)
        ahead_left = self.single_region_density(
            current_direction.left(), edge_length)

        return (ahead_right, back_right, back_left, ahead_left)

    def normalised_regions_density(self, current_direction: Directions,
                                   edge_length: int = 4) -> Tuple[float, float, float, float]:
        """Returns the regions density normalised in the range [-1, 1]"""
        ahead_right, back_right, back_left, ahead_left = self.regions_density(
            current_direction,
            edge_length)
        min = 0
        maxi = edge_length**2
        a = -1
        b = 1
        return (self.normalise_value(ahead_right, min, maxi, a, b),
                self.normalise_value(back_right, min, maxi, a, b),
                self.normalise_value(back_left, min, maxi, a, b),
                self.normalise_value(ahead_left, min, maxi, a, b))

    def single_region_density(
            self, edge_direction: Directions, edge_length: int) -> int:
        """Computes the density of the single region with one edge going from
        head of snake in `edge_direction`direction and the second edge equal to
        `edge_direction.right()`.

        Args:
            edge_direction (Directions): Direction of first edge. Direction of
            second edge is `edge_direction.right()`
            edge_length (int): Length of region edges

        Returns:
            int: Number of free positions in the region
        """
        second_edge_direction: Directions = edge_direction.right()

        # vertex of the region closest to head of snake
        vertex_1: Position = self.snake_positions[0]
        vertex_1 = (vertex_1[0] + edge_direction.value[0],
                    vertex_1[1] + edge_direction.value[1])
        vertex_1 = (vertex_1[0] + second_edge_direction.value[0],
                    vertex_1[1] + second_edge_direction.value[1])

        # vertex of the region farthest from head of snake
        vertex_2 = (
            vertex_1[0] + edge_direction.value[0] * (edge_length),
            vertex_1[1] + edge_direction.value[1] * (edge_length))
        vertex_2 = (
            vertex_2[0] + second_edge_direction.value[0] * (edge_length),
            vertex_2[1] + second_edge_direction.value[1] * (edge_length))

        # computes all positions in the region
        start_horiz: int = vertex_1[0]
        end_horiz: int = vertex_2[0]
        step_horiz: int = 1 if end_horiz >= start_horiz else -1
        start_vert: int = vertex_1[1]
        end_vert: int = vertex_2[1]
        step_vert: int = 1 if end_vert >= start_vert else -1

        region_positions: Set[Position] = set(
            itertools.product(
                range(start_horiz, end_horiz, step_horiz),
                range(start_vert, end_vert, step_vert)))

        # returns size of intersection between free_positions and region_positions
        return len(self.free_positions().intersection(region_positions))
