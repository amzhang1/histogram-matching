import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

img = cv2.imread('test.jpg') # load an image with the original color on it
img2 = cv2.imread('test.jpg') # convert the image to gray scale if
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) # convert the image to gray scale if
# not already a grayscale image
rows, columns, channels = img.shape # row and column of the pixel location
rows1, columns1 = img2.shape # row and column of the pixel location

#initialized 2D arrays of all zeros with a size of 256 elements for grayscale
array1 = np.zeros((256))
array2 = np.zeros((256))

hist,bins = np.histogram(img2.flatten(),256,[0,256])
cdf = hist.cumsum() # cumulative sum of a numpy array

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img3 = cdf[img2]


for e in range(rows1):
	for f in range(columns1):
		pixel_gray1 = img2[e,f]
		array2[pixel_gray1] += 1

# to obtain the histogram of grayscale image, store in array1
for a in range(rows):
	for b in range(columns):
		pixel_gray = img[a,b]
		array1[pixel_gray] += 1

################################################################################
# to plot the image after histogram equalization and making a cumulative histogram with array1
array3 = np.zeros((256))

for index in range(1, len(array2)): # this is to get the cumulative histogram and updating
    # the new array2 with cumulative pixel values from one previous
	array2[index] = array2[index-1] + array2[index]


for c in range(rows1):
	for d in range(columns1):
		pixel_location = img2[c,d]
		img2[c,d] = math.floor((array2[pixel_location] / (rows1 * columns1)) * (len(array2) - 1))


plt.figure(1)
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.plot(array1)

plt.subplot(223)
plt.imshow(img2, 'gray') # to plot the new equalized histogram and show it
plt.subplot(224)
#plt.plot(array2)
#plt.hist(img3.flatten(),256,[0,256], color = 'blue')
plt.hist(img3.flatten(),256,[0,256], color = 'blue')
plt.xlim([0,256])
plt.show()
