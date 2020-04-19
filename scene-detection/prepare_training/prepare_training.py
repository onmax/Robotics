'''
The goal of this class will be to prepare the images to be labelled and be able to use them for the classification model. For that purpose this class offer two functions:
1) Allow user to select frame(s) from the video he wants 
'''

import cv2
import os
import glob
import pygame
import numpy as np
import imageio


class PrepareTraining:
    def __init__(self, video=None, remove_frames=False):
        self.originals_folder = "./images/train/originals"
        self.sections_folder = "./images/train/sections"

        if remove_frames:
            self.remove_files_in(self.remove_files_in)
        if video is not None:
            self.set_frames_from_video(video)
        self.paint_frames()

    '''
    It will remove all the files in the folder
    '''
    def remove_files_in(self, folder):
        files = glob.glob("{}/*".format(folder))
        for f in files:
            os.remove(f)
            print("Removed {}".format(f.split('/')[-1]))

    '''
    It will show to the user a video. This video won't be playing. The user can use left, right 
    arrows to move one second forward or backward or the comma or dot character to move one frame
    forward or backward. Pressing 'Enter' button will save the image in the
    ./images/train/originals/ folder
    '''
    def set_frames_from_video(self, video_name):
        print("Set a frame you want to save typing 'Enter'")
        print("Use left and right arrow to move the video +-1 second")
        print("Use dot and comma buttons to move the video +-1 frame")
        print("Press q to exit the program\n")

        cap = cv2.VideoCapture("./videos/input/{}".format(video_name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame = 0

        while cap.isOpened():
            cap.set(1, current_frame)
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Original video", frame)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    cap.release()
                    break
                elif key == 44:
                    current_frame = max(0, current_frame - 1)
                elif key == 46:
                    current_frame = max(0, current_frame + 1)
                elif key == 81:
                    current_frame = max(0, current_frame - fps)
                elif key == 83:
                    current_frame = max(0, current_frame + fps)
                elif key == 13:
                    img_path = "{}/{}-{}.png".format(self.originals_folder, video_name, int(current_frame))
                    cv2.imwrite(img_path, frame)
                    print("Image saved:", img_path)
            else:
                cap.release()
                break
        cv2.destroyAllWindows()

   
    def roundline(self, srf, color, start, end, radius=1):
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int(start[0]+float(i)/distance*dx)
            y = int(start[1]+float(i)/distance*dy)
            pygame.draw.circle(srf, color, (x, y), radius)
    
    '''
    Shows image img on a window and lets you mark in red, green and blue 
        pixels in the image.
        img: numpy array with the image to be labeled
        radio: is the radio of the circumference used as brush
        returns: a numpy array that is the image painted
    '''
    def select_fg_bg(self, img, radio=2):
        # Creates the screen where the image will be displayed
        # Shapes are reversed in img and pygame screen
        screen = pygame.display.set_mode(img.shape[-2::-1])

        # imgpyg=pygame.image.load(imgName)
        imgpyg = pygame.image.frombuffer(img, img.shape[-2::-1], 'RGB')
        screen.blit(imgpyg, (0, 0))
        pygame.display.flip()  # update the display

        draw_on = False
        last_pos = (0, 0)
        color_red = (255, 0, 0)
        color_green = (0, 255, 0)
        color_blue = (0, 0, 255)

        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                break
            if e.type == pygame.KEYDOWN and e.unicode == 'z':
                return None, True
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
                    self.roundline(screen, color, e.pos, last_pos,  radio)
                last_pos = e.pos
            pygame.display.flip()

        imgOut = np.ndarray(
            shape=img.shape[:2]+(4,), dtype='u1', buffer=screen.get_buffer().raw)
        pygame.quit()

        return cv2.cvtColor(imgOut[:, :, :3], cv2.COLOR_BGR2RGB), False

    def paint_frames(self):
        '''
        It will show the user all the frames in the ./images/training/originals frame and allow to the user to label them. 
        '''

        print("Select pixels you want to label. Use the three buttons of your mouse.\nControls:\n\t'z':to start again to paint all the pixels.\n\t'q':to finish and save the image.")

        # For each frame, we will show it wo the user
        originals_files = glob.glob("{}/*".format(self.originals_folder))
        for p in originals_files:
            original = imageio.imread(p)
            while True:
                # The user can draw on the frame to label the pixels
                sections, start_again = self.select_fg_bg(original)
                if not start_again:
                    break
            cv2.cvtColor(sections, cv2.COLOR_BGR2RGB)
            img_path = "{}/{}".format(self.sections_folder, p.split('/')[-1])
            imageio.imsave(img_path, sections)
