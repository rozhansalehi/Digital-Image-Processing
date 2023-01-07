import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

################# Part A ###################
cube=cv.imread('Cube.tif', 1)

################# Part B ###################
gray_cube = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
cv.imshow('fig2',gray_cube)
k=cv.waitKey(0)
if k==ord("o"):
    cv.imwrite('gray cube.png',gray_cube)

print('shape of gray cube:',gray_cube.shape,'\n','data type of gray cube:',gray_cube.dtype)

################# Part C ###################
def Intensity(old_img, n):
    new_img = old_img//(old_img.max()/(2**n-1))
    new_img=new_img.astype('uint8')
    return new_img

################# Part D ###################
input_img=cv.imread('gray cube.png',0)

plt.figure(1)
plt.subplot(231), plt.imshow(Intensity(input_img,8),'gray'), plt.title('8 BiTs'), plt.xticks([]),plt.yticks([])
plt.subplot(232), plt.imshow(Intensity(input_img,5),'gray'), plt.title('5 BiTs'), plt.xticks([]),plt.yticks([])
plt.subplot(233), plt.imshow(Intensity(input_img,3),'gray'), plt.title('3 BiTs'), plt.xticks([]),plt.yticks([])
plt.subplot(234), plt.imshow(Intensity(input_img,2),'gray'), plt.title('2 BiTs'), plt.xticks([]),plt.yticks([])
plt.subplot(235), plt.imshow(Intensity(input_img,1),'gray'), plt.title('1 BiT'), plt.xticks([]),plt.yticks([])


################# Part E&F ###################
ret,binary_cube=cv.threshold(gray_cube,127,255,cv.THRESH_BINARY)
plt.subplot(236), plt.imshow(binary_cube,'gray'), plt.title('Binary'), plt.xticks([]),plt.yticks([])
plt.suptitle('gray cube with different levels of intensity')
plt.savefig('9733045-3.png')
plt.show()