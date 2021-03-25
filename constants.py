# windows
WIDTH, HEIGHT = (500, 500)
BACKGROUND_COLOR = (240, 240, 240)
CAPTION = "Snake"

# snake
SNK_HEAD_COLOR = (220, 220, 20)
SNK_COLOR = (100, 220, 100)
SNK_RADIUS = 10
SNK_SPEED = SNK_RADIUS / 2  # pixels per frame
SNK_INIT_POS = (int(WIDTH / 2) - 1, int(HEIGHT / 2) - 1)
SNK_DIR = {"UP": (0, -1), "DOWN": (0, 1), "RIGHT": (1, 0), "LEFT": (-1, 0)}

# apple
APL_COLOR = (255, 50, 50)
APL_RADIUS = 10
APL_UPDATE_RATE = 5  # seconds

# general
FPS = 30
GRID_SIZE = SNK_RADIUS * 2
