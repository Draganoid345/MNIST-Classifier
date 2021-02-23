import pygame
import tensorflow as tf
import numpy as np
from PIL import Image



WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('MNIST')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Spot:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = BLACK
        self.width = width

    def get_pos(self):
        return self.row, self.col

    def make_white(self):
        self.color = WHITE

    def make_black(self):
        self.color = BLACK

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))


def make_grid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap)
            grid[i].append(spot)

    return grid


def draw(win, grid, rows, width):
    win.fill(BLACK)
    for row in grid:
        for spot in row:
            spot.draw(win)



def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x, y = pos

    row = x // gap
    col = y // gap

    return row, col


def main(win, width):
    ROWS = 28
    grid = make_grid(ROWS, width)
    pred = -1
    run = True
    pygame.init()

    while run:
        draw(win, grid, ROWS, width)
        if pred != -1:
            win.blit(text, textRect)
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.make_white()
                # grid[row-1][col].make_white()
                # grid[row+1][col].make_white()
                # grid[row][col+1].make_white()
                # grid[row][col-1].make_white()
                # grid[row-1][col-1].make_white()
                # grid[row+1][col+1].make_white()
                # grid[row-1][col+1].make_white()
                # grid[row+1][col-1].make_white()

            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.make_black()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    grid = make_grid(ROWS, width)
                    pred = -1

                if event.key == pygame.K_SPACE:
                    model = tf.keras.models.load_model('testmodel.model')
                    pygame.image.save(win, 'test.jpeg')
                    img = Image.open('test.jpeg').convert('L').resize((28, 28), Image.ANTIALIAS)
                    img = np.array(img)
                    pred = model.predict(img[None, :, :])
                    pred = str(np.argmax(pred))

                    font = pygame.font.Font('freesansbold.ttf', 100)
                    text = font.render(pred, True, WHITE, BLACK)
                    textRect = text.get_rect()
                    textRect.center = (WIDTH // 2, WIDTH -100)
                    win.blit(text, textRect)
            pygame.display.update()


main(WIN, WIDTH)