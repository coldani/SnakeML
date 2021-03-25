import pygame
import random
import itertools


class Windows:
    def __init__(self, width, height, background_color, caption):
        self.width = width
        self.height = height
        self.background_color = background_color
        self.caption = caption

        self.window = pygame.display.set_mode((width, height))
        self.window.fill(background_color)
        pygame.display.set_caption(caption)

    def update(self):
        self.window.fill(self.background_color)


class Snake:
    def __init__(
        self, window, color, head_color, radius, speed, initial_position
    ):
        self.window = window
        self.color = color
        self.head_color = head_color
        self.radius = radius
        self.speed = speed
        self.positions = [initial_position]
        self.directions = [None]
        self.score = 0
        self.hit = False

        self.update_rate = int(2 * radius / speed)
        self.cycle_count = 0

        self.snake_len = 1
        self.growth_cycle_handler = {}
        self.growth_flag = False
        self.new_positions = {}
        self.new_directions = {}

    def update(self, new_direction):

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
            if (
                next(self.growth_cycle_handler[self.snake_len + 1])
                == self.update_rate - 1
            ):
                self.positions.append(self.new_positions[self.snake_len + 1])

                del self.growth_cycle_handler[self.snake_len + 1]
                del self.new_positions[self.snake_len + 1]
                self.snake_len += 1

                if len(self.growth_cycle_handler) == 0:
                    self.growth_flag = False

        # redraw the snake
        for i, position in enumerate(self.positions):
            if i == 0:
                color = self.head_color
            else:
                color = self.color
            pygame.draw.circle(self.window.window, color, position, self.radius)

    def check_eat(self, apple):
        apl_x, apl_y = apple.center
        apl_radius = apple.radius
        head_position = self.positions[0]

        # yumm!
        if (
            (head_position[0] - self.radius <= apl_x + apl_radius)
            and (head_position[0] + self.radius >= apl_x - apl_radius)
            and (head_position[1] - self.radius <= apl_y + apl_radius)
            and (head_position[1] + self.radius >= apl_y - apl_radius)
        ):
            self.score += 1
            print(self.score)
            self.grow()
            return True
        else:
            return False

    def check_hit(self):
        head_position = self.positions[0]

        # hit the border
        if (
            (head_position[0] - self.radius <= 0)
            or (head_position[0] + self.radius >= self.window.width)
            or (head_position[1] - self.radius <= 0)
            or (head_position[1] + self.radius >= self.window.height)
        ):
            self.hit = True

        # hit the body
        if self.snake_len > 2:
            for position in self.positions[2:]:
                if (
                    (head_position[0] - self.radius < position[0] + self.radius)
                    and (
                        head_position[0] + self.radius
                        > position[0] - self.radius
                    )
                    and (
                        head_position[1] - self.radius
                        < position[1] + self.radius
                    )
                    and (
                        head_position[1] + self.radius
                        > position[1] - self.radius
                    )
                ):
                    self.hit = True

    def grow(self):
        self.growth_cycle_handler[self.snake_len + 1] = iter(
            range(self.update_rate)
        )
        self.new_positions[self.snake_len + 1] = self.positions[-1]
        self.directions.extend([None for x in range(self.update_rate)])
        self.growth_flag = True


class Apple:
    def __init__(self, window, color, radius):
        self.window = window
        self.color = color
        self.radius = radius
        self.center = self.recenter()
        self.recenter_count = 0

    def update(self, recenter):
        if recenter:
            self.center = self.recenter()
        pygame.draw.circle(
            self.window.window, self.color, self.center, self.radius
        )

    def recenter(self):
        x = random.randint(self.radius, self.window.width - self.radius)
        y = random.randint(self.radius, self.window.height - self.radius)
        return (x, y)
