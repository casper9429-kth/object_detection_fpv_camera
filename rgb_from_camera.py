#!/usr/bin/env python
import cv2
import numpy as np
import time


# Function to take a picture from the camera
def take_picture(camera_device="/dev/video0"):
    """
    Take picture from the camera
    To determine the camera device, run the following command in the terminal:
    v4l2-ctl --list-devices
    This command will list all the devices connected to the computer.
    For laptop, the camera device is usually /dev/video0
    For the NUC, the arm camera device is usually /dev/video6
    Will return the image as a numpy array in BGR format uint8
    """
    # v4l2-ctl --list-devices
    cap = cv2.VideoCapture(camera_device)
    ret, frame = cap.read()
    cv2.imwrite("image.png", frame)
    return frame


def main():
    #node = CameraNode()
    #node.start()
    frame = take_picture()



main()

