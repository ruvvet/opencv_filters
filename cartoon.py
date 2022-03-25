
import cv2 as cv
import numpy as np
import utils

def cartoon(imageFile):

    image = utils.create_image(imageFile)

    #convert to gray scale
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #apply gaussian blur
    grayImage = cv.GaussianBlur(grayImage, (3, 3), 0)

    #detect edges
    edgeImage = cv.Laplacian(grayImage, -1, ksize=5)
    edgeImage = 255 - edgeImage

    #threshold image
    ret, edgeImage = cv.threshold(edgeImage, 150, 255, cv.THRESH_BINARY)

    #blur images heavily using edgePreservingFilter
    edgePreservingImage = cv.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.4)

    #create output matrix
    output =np.zeros(grayImage.shape)

    #combine cartoon image and edges image
    output = cv.bitwise_and(edgePreservingImage, edgePreservingImage, mask=edgeImage)


    retval, buffer = cv.imencode('.jpg', output)

    return buffer


