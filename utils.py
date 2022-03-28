import cv2 as cv
import numpy as np


def create_image(image_buffer):

    image = cv.imdecode(np.fromstring(image_buffer, dtype=np.uint8), cv.IMREAD_COLOR)

    resized_image = cv.resize(image, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)
    return resized_image


