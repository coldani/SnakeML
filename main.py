import pygame
import itertools
from objects import *
from constants import *

pygame.init()
win = Windows(WIDTH, HEIGHT, BACKGROUND_COLOR, CAPTION)
pygame.display.flip()

snake = Snake(
    win, SNK_COLOR, SNK_HEAD_COLOR, SNK_RADIUS, SNK_SPEED, SNK_INIT_POS
)
apple = Apple(win, APL_COLOR, APL_RADIUS)
grid_cycle = itertools.cycle(range(int(GRID_SIZE / SNK_SPEED)))


def update(apl_recenter, snk_direction):

    win.update()
    apple.update(apl_recenter)
    snake.update(snk_direction)
    if snake.check_eat(apple):
        apple.update(True)
        apple.recenter_count = 0
    pygame.display.flip()
    snake.check_hit()


running = True
snk_direction = SNK_DIR["UP"]
key = None

while running:
    pygame.time.Clock().tick(FPS)
    apl_recenter = False
    apple.recenter_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            key = event.key

    new_direction = snk_direction
    count_cycle = next(grid_cycle)
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
    if apple.recenter_count >= (APL_UPDATE_RATE * FPS):
        apl_recenter = True
        apple.recenter_count = 0
    update(apl_recenter, snk_direction)
    if snake.hit:
        running = False

