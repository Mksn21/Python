import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *

def plot_function():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Draw a simple function (sin(x) in this case)
    glBegin(GL_LINES)
    for x in range(-100, 100):
        y = 50 * math.sin(x / 10.0)
        glVertex2f(x, y)
    glEnd()

    pygame.display.flip()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        plot_function()

if __name__ == "__main__":
    main()
