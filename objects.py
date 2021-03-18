import pygame
import random


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
    def __init__(self, window, color, radius, speed, initial_position):
        self.window = window
        self.color = color
        self.radius = radius
        self.speed = speed
        self.position = initial_position
        self.score = 0
        self.hit = False

    def update(self, direction):
        x, y = self.position
        x += direction[0] * self.speed
        y += direction[1] * self.speed
        self.position = (x, y)
        pygame.draw.circle(self.window.window, self.color, self.position, self.radius)

    def check_eat(self, apple):
        apl_x, apl_y = apple.center
        apl_radius = apple.radius

        if (
            (self.position[0] - self.radius <= apl_x + apl_radius)
            and (self.position[0] + self.radius >= apl_x - apl_radius)
            and (self.position[1] - self.radius <= apl_y + apl_radius)
            and (self.position[1] + self.radius >= apl_y - apl_radius)
        ):
            self.score += 1
            print(self.score)
            self.grow()

            return True
        else:
            return False

    def check_hit(self):

        # borders
        if (
            (self.position[0] - self.radius <= 0)
            or (self.position[0] + self.radius >= self.window.width)
            or (self.position[1] - self.radius <= 0)
            or (self.position[1] + self.radius >= self.window.height)
        ):
            self.hit = True

    def grow(self):
        pass


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
        pygame.draw.circle(self.window.window, self.color, self.center, self.radius)

    def recenter(self):
        x = random.randint(self.radius, self.window.width - self.radius)
        y = random.randint(self.radius, self.window.height - self.radius)
        return (x, y)
