import numpy as np
import cv2 as cv
import utils


def clarendon(imageFile):
    
    image = utils.create_image(imageFile)

    #create a copy of input image to work on
    clarendon = image.copy()

    #split the channels
    blueChannel, greenChannel, redChannel = cv.split(clarendon)

    #Interpolation values
    originalValues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
    blueValues =     np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255 ])
    redValues =      np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249 ])
    greenValues =    np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])

    #Creating the lookuptables
    fullRange = np.arange(0,256)
    #Creating the lookuptable for blue channel
    blueLookupTable = np.interp(fullRange, originalValues, blueValues )
    #Creating the lookuptables for green channel
    greenLookupTable = np.interp(fullRange, originalValues, greenValues )
    #Creating the lookuptables for red channel
    redLookupTable = np.interp(fullRange, originalValues, redValues )

    #Apply the mapping for blue channel
    blueChannel = cv.LUT(blueChannel, blueLookupTable)
    #Apply the mapping for green channel
    greenChannel = cv.LUT(greenChannel, greenLookupTable)
    #Apply the mapping for red channel
    redChannel = cv.LUT(redChannel, redLookupTable)

    #merging back the channels
    output = cv.merge([blueChannel, greenChannel, redChannel])

    #convert to uint8
    output = np.uint8(output)

    retval, buffer = cv.imencode('.jpg', output)


    return buffer