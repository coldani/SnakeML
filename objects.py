import pygame
import random
import itertools
from constants import *


class Windows:
    def __init__(self):
        self.width = WIDTH * GRID_SIZE
        self.height = HEIGHT * GRID_SIZE
        self.grid_size = GRID_SIZE
        self.background_color = BACKGROUND_COLOR
        self.caption = CAPTION

        self.window = pygame.display.set_mode((self.width, self.height))
        self.window.fill(self.background_color)
        pygame.display.set_caption(self.caption)

    def update(self):
        self.window.fill(self.background_color)


class Snake:
    def __init__(self, window):
        self.window = window
        self.color = SNK_COLOR
        self.head_color = SNK_HEAD_COLOR
        self.size = GRID_SIZE
        self.speed = SNK_SPEED
        self.positions = [SNK_INIT_POS]
        self.directions = [SNK_INIT_DIR]
        self.score = 0

        self.update_rate = self.size // self.speed
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
            if next(self.growth_cycle_handler[self.snake_len + 1]) == self.update_rate - 1:
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
            body_piece = pygame.Rect(position, (self.size, self.size))
            pygame.draw.rect(self.window.window, color, body_piece, 0)
            pygame.draw.rect(self.window.window, (0, 0, 0), body_piece, 1)

    def check_eat(self, apple):
        apl_x, apl_y = apple.center
        apl_radius = apple.radius
        head_position = self.positions[0]

        # yumm!
        if (
            (head_position[0] < apl_x + apl_radius)
            and (head_position[0] + self.size > apl_x - apl_radius)
            and (head_position[1] < apl_y + apl_radius)
            and (head_position[1] + self.size > apl_y - apl_radius)
        ):
            self.score += 1
            print(self.score)
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
            or (head_position[0] + self.size > self.window.width)
            or (head_position[1] < 0)
            or (head_position[1] + self.size > self.window.height)
        ):
            hit = True

        # hit the body
        if self.snake_len > 2:
            for position in self.positions[2:]:
                if (
                    (head_position[0] < position[0] + self.size)
                    and (head_position[0] + self.size > position[0])
                    and (head_position[1] < position[1] + self.size)
                    and (head_position[1] + self.size > position[1])
                ):
                    hit = True
        return hit

    def grow(self):
        self.growth_cycle_handler[self.snake_len + 1] = iter(range(self.update_rate))
        self.new_positions[self.snake_len + 1] = self.positions[-1]
        self.directions.extend([None for x in range(self.update_rate)])
        self.growth_flag = True


class Apple:
    def __init__(self, window):
        self.window = window
        self.color = APL_COLOR
        self.radius = APL_RADIUS
        self.center = self.recenter()
        self.recenter_count = 0

    def update(self, recenter):
        if recenter:
            self.center = self.recenter()
        pygame.draw.circle(self.window.window, self.color, self.center, self.radius, 0)
        pygame.draw.circle(self.window.window, (0, 0, 0), self.center, self.radius, 1)

    def recenter(self):
        grid_size = self.window.grid_size
        grid_w = self.window.width // grid_size - 1
        grid_h = self.window.height // grid_size - 1
        x = random.randint(0, grid_w) * grid_size + self.radius
        y = random.randint(0, grid_h) * grid_size + self.radius
        return (x, y)


class Game:
    def __init__(self, window, snake, apple):
        self.window = window
        self.snake = snake
        self.apple = apple
        self.grid_cycle = itertools.cycle(range(self.window.grid_size // self.snake.speed))
        self.running = True

    def update(self, apl_recenter, snk_direction):
        self.window.update()
        self.apple.update(apl_recenter)
        self.snake.update(snk_direction)
        if self.snake.check_eat(self.apple):
            self.apple.update(True)
            self.apple.recenter_count = 0
        pygame.display.flip()

    def run(self):
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

            self.update(apl_recenter, snk_direction)

            if self.snake.check_hit():
                running = False
                print("dead")
