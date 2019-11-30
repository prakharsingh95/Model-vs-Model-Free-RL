import cv2

import settings

class VideoLogger(object):

    def __init__(self, file, height, width):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(f'{settings.VIDEO_LOG_DIR}/{file}.avi', fourcc,
                              24.0, (width, height))

    def write(self, obs):
        self.out.write(obs)

    def __del__(self):
        self.out.release()