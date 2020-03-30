#######################################################################
# This code lets you paint on top of an image and returns the painted image
# it can be used to select pixels somehow in an image
#
# It requires that you install "python-pygame" and "python-opencv"
#
# The interesting funcion here is "select_fg_bg" read documentation below
#######################################################################
import pygame
import numpy as np
import cv2


def roundline(srf, color, start, end, radius=1):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0]+float(i)/distance*dx)
        y = int(start[1]+float(i)/distance*dy)
        pygame.draw.circle(srf, color, (x, y), radius)


img_base = None


def setImage(screen):
    # imgpyg=pygame.image.load(imgName)
    imgpyg = pygame.image.frombuffer(img_base, img_base.shape[-2::-1], 'RGB')
    screen.blit(imgpyg, (0, 0))
    # update the display
    pygame.display.flip()


def select_fg_bg(img, radio=2):
    global img_base

    """ Shows image img on a window and lets you mark in red, green and blue 
        pixels in the image.
        img: numpy array with the image to be labeled
        radio: is the radio of the circumference used as brush
        returns: a numpy array that is the image painted
    """
    img_base = img

    # Creates the screen where the image will be displayed
    # Shapes are reversed in img and pygame screen
    screen = pygame.display.set_mode(img.shape[-2::-1])
    setImage(screen)

    draw_on = False
    last_pos = (0, 0)
    color_red = (255, 0, 0)
    color_green = (0, 255, 0)
    color_blue = (0, 0, 255)

    finish = False

    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT or e.type == pygame.KEYDOWN and e.key == 13:
            break
        if e.type == pygame.KEYDOWN and e.unicode == 'q':
            finish = True
            break
        if e.type == pygame.KEYDOWN and e.unicode == 'z':
            draw_on = False
            setImage(screen)
        if e.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                color = color_red
            elif pygame.mouse.get_pressed()[2]:
                color = color_green
            else:
                color = color_blue
            pygame.draw.circle(screen, color, e.pos, radio)
            draw_on = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radio)
                roundline(screen, color, e.pos, last_pos,  radio)
            last_pos = e.pos
        pygame.display.flip()

    imgOut = np.ndarray(
        shape=img.shape[:2]+(4,), dtype='u1', buffer=screen.get_buffer().raw)
    pygame.quit()

    return cv2.cvtColor(imgOut[:, :, :3], cv2.COLOR_BGR2RGB), finish
