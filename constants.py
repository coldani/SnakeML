# general
FPS = 10

# windows
WIDTH, HEIGHT = (500, 500)
BACKGROUND_COLOR = (120, 140, 50)
CAPTION = "Snake"

# snake
SNK_COLOR = (0, 255, 0)
SNK_SPEED = 10
SNK_RADIUS = 10
SNK_INIT_POS = (int(WIDTH / 2), int(HEIGHT / 2))
SNK_DIR = {"UP": (0, -1), "DOWN": (0, 1), "RIGHT": (1, 0), "LEFT": (-1, 0)}

# apple
APL_COLOR = (255, 0, 0)
APL_RADIUS = 10
APL_UPDATE_RATE = 5  # seconds
