import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import numpy as np

def plot_function():
    # Create a simple function (sin(x) in this case)
    x = np.arange(-100, 100, 0.1)
    y = 50 * np.sin(x / 10.0)

    # Plot the function using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(x, y)

    # Convert Matplotlib figure to a Pygame surface
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    surf = pygame.image.load(io.BytesIO(buf.read()))

    return surf

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        plot_surface = plot_function()

        screen = pygame.display.set_mode(display)
        screen.blit(plot_surface, (0, 0))
        pygame.display.flip()

        clock.tick(60)

if __name__ == "__main__":
    main()
