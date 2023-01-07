import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#################### Part A ###################
HeadCT = cv.imread('HeadCT.tif', 0)
ret, HeadCT_binary = cv.threshold(HeadCT, 100, 255, cv.THRESH_BINARY)

plt.figure(1)
plt.imshow(HeadCT_binary, cmap='gray', vmin=0, vmax=255)
plt.axis(False)
plt.savefig('HeadCT binary.png')
#################### Part B ###################
kernel= cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20)) # elliptic kernel
closing = cv.morphologyEx(HeadCT_binary, cv.MORPH_CLOSE, kernel)

plt.figure(2)
plt.imshow(closing, cmap='gray', vmin=0, vmax=255)
plt.axis(False)
plt.savefig('HeadCT closing.png')
#################### Part C ###################
def hole_filling(img, seed):
    x_k_1 = np.full_like(img,0)
    x_k_1[seed] = 255
    kernel = np.array([[1 , 1 , 1],
                       [1 , 1 , 1],
                       [1 , 1 , 1]])
    while 1:
        dilation = cv.dilate(x_k_1, kernel, iterations=1)
        img_c = np.bitwise_not(img) 
        x_k = dilation & img_c
        if (x_k == x_k_1).all():
            break
        else:
            x_k_1 = x_k
    return (x_k | img)

seed1 = (162,427)
img1 = hole_filling(closing, seed1)

seed2 = (166,284)
img2 = hole_filling(img1, seed2)

seed3 = (82,247)
img3 = hole_filling(img2, seed3)

plt.figure(3)
plt.subplot(131), plt.imshow(img1, cmap='gray', vmin=0, vmax=255), plt.axis(False), plt.title('1')
plt.subplot(132), plt.imshow(img2, cmap='gray', vmin=0, vmax=255), plt.axis(False), plt.title('2')
plt.subplot(133), plt.imshow(img3, cmap='gray', vmin=0, vmax=255), plt.axis(False), plt.title('3')
plt.savefig('hole filling.png')
#################### Part D ###################
kernel2 =  np.array([[1 , 1 , 1],
                    [1 , 1 , 1],
                    [1 , 1 , 1]])
boundary = img3 - (cv.erode(img3, kernel2, iterations=1))         
plt.figure(4)
plt.imshow(boundary, cmap='gray', vmin=0, vmax=255) 
plt.title('finding boundaries')
plt.axis(False)    
plt.savefig('finding boundaries.png')      
plt.show()