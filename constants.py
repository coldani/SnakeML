# general
FPS = 8

# window
WIDTH, HEIGHT = (30, 30)
GRID_SIZE = 15
BACKGROUND_COLOR = (240, 240, 240)
CAPTION = "Snake"

# snake
SNK_HEAD_COLOR = (220, 220, 20)
SNK_COLOR = (100, 220, 100)
SNK_SPEED = 15  # pixels per frame
SNK_INIT_POS = ((WIDTH // 2) * GRID_SIZE, (HEIGHT // 2) * GRID_SIZE)
SNK_DIR = {"UP": (0, -1), "DOWN": (0, 1), "RIGHT": (1, 0), "LEFT": (-1, 0)}
SNK_INIT_DIR = SNK_DIR["UP"]

# apple
APL_COLOR = (255, 50, 50)
APL_RADIUS = GRID_SIZE / 2
APL_UPDATE_RATE = 50  # seconds

