import numpy as np
import cv2 as cv
import utils

from matplotlib import pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import colorsys


def get_dominant_color(image, k=4, image_processing_size = None ):

    if image_processing_size is not None:
        image = cv.resize(image, image_processing_size, 
                                interpolation = cv.INTER_AREA)

    # convert to hsv
    cv.cvtColor(image, cv.COLOR_BGR2HSV)

    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
 
    labels = clt.fit_predict(image)
    print(labels)

    #count labels to find most popular
    label_counts = Counter(labels)
 
    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    print(dominant_color)

    return  {'hsv': tuple(dominant_color), 'rgb':colorsys.hsv_to_rgb(*dominant_color)}


def color_range():
    return





def colorpop(imageFile):

    image = utils.create_image(imageFile)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    dominant_color=get_dominant_color(image)

    return dominant_color


    cv.imshow('hi', image)
    
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv.waitKey(0) 

    # dominant_color_hsv = dominant_color['hsv']

    # lower_red = np.array([dominant_color_hsv[0]-10,dominant_color_hsv[1]-10, dominant_color_hsv[2]-40] )
    # upper_red = np.array([dominant_color_hsv[0]+10,dominant_color_hsv[1]+10, dominant_color_hsv[2]+40])

    # #create a mask using the bounds set
    # mask = cv.inRange(hsv, lower_red, upper_red)
    # #create an inverse of the mask
    # mask_inv = cv.bitwise_not(mask)
    # #Filter only the red colour from the original image using the mask(foreground)
    # res = cv.bitwise_and(image, image, mask=mask)
    # #Filter the regions containing colours other than red from the grayscale image(background)
    # background = cv.bitwise_and(gray, gray, mask = mask_inv)
    # #convert the one channelled grayscale background to a three channelled image
    # background = np.stack((background,)*3, axis=-1)
    # #add the foreground and the background
    # added_img = cv.add(res, background)


    # retval, buffer = cv.imencode('.jpg', added_img)

    return buffer






