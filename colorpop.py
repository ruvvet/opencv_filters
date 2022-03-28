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
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
       #reshape the image to be a list of pixels
    image_px_list = image.reshape((image.shape[0] * image.shape[1], 3))
   
    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image_px_list)
 
    #count labels to find most popular
    label_counts = Counter(labels)
 
    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    dominant_color_convert = convert_cv_hsv_values_to_hsv(*dominant_color)

    return  {'hsv': dominant_color_convert, 'rgb':colorsys.hsv_to_rgb(*dominant_color)}
  

def convert_cv_hsv_values_to_hsv(h,s,v):
    return ((h/179)*360, (s/255)*100, (v/255)*100)


def get_dominant_hue_range(h,s,v):
    return {'upper': [h-10,s-10, v-40 ], 'lower':[h+10,s+10, v+40 ] }




def colorpop(imageFile):

    image = utils.create_image(imageFile)
    print(image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    dominant_color=get_dominant_color(image)
    # dominant_hsv_range = get_dominant_hue_range(*dominant_color['hsv'])
    dominant_hsv_range = get_dominant_hue_range(0,255,255)

    # lower = np.array(dominant_hsv_range['lower'])
    # upper = np.array(dominant_hsv_range['upper'])

    lower = np.array([160,100,20])
    upper = np.array([179,255,255])

    #create a mask using the bounds set
    mask = cv.inRange(hsv, lower, upper)
    #create an inverse of the mask
    mask_inv = cv.bitwise_not(mask)
    #Filter only the red colour from the original image using the mask(foreground)
    res = cv.bitwise_and(image, image, mask=mask)
    #Filter the regions containing colours other than red from the grayscale image(background)
    background = cv.bitwise_and(gray, gray, mask = mask_inv)
    #convert the one channelled grayscale background to a three channelled image
    background = np.stack((background,)*3, axis=-1)
    #add the foreground and the background
    added_img = cv.add(res, background)


    retval, buffer = cv.imencode('.jpg', added_img)

    return buffer






