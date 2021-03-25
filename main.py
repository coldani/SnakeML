import pygame
import itertools
from objects import *
from constants import *

pygame.init()
win = Windows()

snake = Snake(win)
apple = Apple(win)

game = Game(win, snake, apple)
game.run()
