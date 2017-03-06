import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# to read the two images here and convert to grayscale
img = cv2.imread('day.jpg') # load an image with the original color on it
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert the color image to gray scale

img2 = cv2.imread('night.jpg', 1) # load an image with the original color on it
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) # convert the color image to gray scale

# initialize two 1D array of size 256 elements for both pictures day and night
array1 = np.zeros((256))
array2 = np.zeros((256))

rows1, columns1 = img.shape # pixel location determined by row x column
rows2, columns2 = img2.shape # pixel location of the second image by row x column

# next two double for loops is to determine the histograms for each picture img and img2
# which are the histograms for day and night stored in array1 and array2
for i in range(rows1):
    for j in range(columns1):
         pixel = img[i,j] # obtain the pixel value of that at location i,j
         array1[pixel] += 1 # increment the histogram[pixel] location by 1

for a in range(rows2):
    for b in range(columns2):
        pixel1 = img2[a,b] # obtain the pixel value of that at location a,b
        array2[pixel1] += 1 # increment the histogram[pixel2] location by 1

#plt.plot(array1) # using this for testing if my histograms are correct
#plt.plot(array2)
#plt.xlim([0,256])
#plt.show() # basically to plot array1 and array2

# this is to get the total sum of each array1 and array2 separately, we need this
# to do the next step to normalize the histogram
arr_sum1 = int(math.floor(np.sum(array1)))
arr_sum2 = int(math.floor(np.sum(array2)))

# normalize each histogram (i.e. array1 and array2 are histograms, so all the
# values in both arrays now are divided by there total sum therefore obtaining decimals)
for item in range(len(array1)):
    array1[item] = array1[item] / arr_sum1

for items in range(len(array2)):
    array2[items] = array2[items] / arr_sum2

# using the Bhattacharya Coefficient formula to calculate 1 for a match, 0 for not
# a match, but closely resembling each other is a high decimal number
array3 = np.zeros((256)) # use this container to see the sum of array1 and array2
for index in range(256):
    array3[index] = math.sqrt(array1[index] * array2[index])

#b_coefficient = math.floor(np.sum(array3))
b_coefficient = np.sum(array3)
print "The Bhattacharya Coefficient is " + str(b_coefficient)
