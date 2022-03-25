from cv2 import resize
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from collections import Counter
from sklearn.cluster import KMeans
import colorsys


# def blur(img):

#     im = cv.imread(img)
#     rows, cols = im.shape[:2]
#     # Create a Gaussian filter
#     kernel_x = cv.getGaussianKernel(cols,800)
#     kernel_y = cv.getGaussianKernel(rows,800)
#     kernel = kernel_y * kernel_x.T
#     filter = 255 * kernel / np.linalg.norm(kernel)
#     blurred_image = np.copy(im)
#     # for each channel in the input image, we will apply the above filter
#     for i in range(3):
#         blurred_image[:,:,i] = blurred_image[:,:,i] * filter

#     return blurred_image



def create_image():
    path = 'dexter.jpg'
    image = cv.imread(path)
    resized_image = cv.resize(image, None, fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)
    return resized_image


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

    #count labels to find most popular
    label_counts = Counter(labels)
 
    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return  {'hsv': tuple(dominant_color), 'rgb':colorsys.hsv_to_rgb(*dominant_color)}





def test():

    image = create_image()

    alpha = 2.0 # contrast
    beta = 0    # brightness

    dominant_color= get_dominant_color(image, 4, (25,25))

    contrast = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    hsv = cv.cvtColor(contrast, cv.COLOR_BGR2HSV)

    threshold = 70

    lower_bound = np.array([x-threshold for x in dominant_color['hsv']])
    upper_bound = np.array([x+threshold for x in dominant_color['hsv']])
  


    saturation = cv.inRange(hsv, lower_bound, upper_bound )  

    res = cv.bitwise_and(image,image, saturation)

    dst = cv.addWeighted(res,0.5, contrast, 0.5, 0.0)

    # cv.imshow('omage', resized_image)
    # cv.imshow('contrast', contrast)
    # cv.imshow('res', res)
    cv.imshow('a', dst)
    cv.waitKey(0)
    