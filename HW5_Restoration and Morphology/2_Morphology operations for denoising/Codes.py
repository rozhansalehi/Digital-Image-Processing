import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#################### Part A&B ###################
img = cv.imread('fingerprint.tif', 0)
ret,binary_img = cv.threshold(255-img, 120, 255, cv.THRESH_BINARY)
plt.figure(1)
plt.imshow(binary_img, cmap='gray', vmin=0, vmax=255)
plt.axis(False)
plt.savefig('finger print binary img.png')
#################### Part C ###################
kernel = np.uint8([[0 , 1 , 0],
                   [1 , 1 , 1],
                   [0 , 1 , 0]])
                   
opening = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(binary_img, cv.MORPH_CLOSE, kernel)
plt.figure(2)
plt.subplot(121), plt.imshow(opening, cmap='gray', vmin=0,vmax=255)
plt.title('opening')
plt.axis(False)
plt.subplot(122), plt.imshow(closing, cmap='gray', vmin=0,vmax=255)
plt.title('closing')
plt.axis(False)
plt.savefig('opening & closing.png')
#################### Part E ###################
closing_of_opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

plt.figure(3)
plt.imshow(closing_of_opening , cmap='gray', vmin=0, vmax=255)
plt.title('closing of opening')
plt.axis(False)
plt.savefig('enhanced img.png')
plt.show()