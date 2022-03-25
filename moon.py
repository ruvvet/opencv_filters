
import cv2 as cv
import utils
import numpy as np

def moon(imageFile):

    #Read input image
    image = utils.create_image(imageFile)

    #create a clone of input image to work on
    output = image.copy()

    #convert to LAB color space
    output = cv.cvtColor(output, cv.COLOR_BGR2LAB)

    #split into channels
    L, A, B = cv.split(output)

    #Interpolation values
    originalValues = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255 ])
    values =         np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255  ])

    #create lookup table
    allValues = np.arange(0, 256)

    #Creating the lookuptable
    lookuptable = np.interp(allValues, originalValues, values)

    #apply mapping for L channels
    L = cv.LUT(L, lookuptable)

    #convert to uint8
    L = np.uint8(L)

    #merge back the channels
    output = cv.merge([L, A, B])

    #convert back to BGR color space
    output = cv.cvtColor(output, cv.COLOR_LAB2BGR)

    #desaturate the image
    #convert to HSV color space
    output = cv.cvtColor(output, cv.COLOR_BGR2HSV)

    #split into channels
    H, S, V = cv.split(output)

    #Multiply S channel by saturation scale value
    S = S * 0.01

    #convert to uint8
    S = np.uint8(S)

    #limit the values between 0 and 256
    S = np.clip(S, 0, 255)

    #merge back the channels
    output = cv.merge([H, S, V])

    #convert back to BGR color space
    output = cv.cvtColor(output, cv.COLOR_HSV2BGR)

    retval, buffer = cv.imencode('.jpg', output)

    return buffer
