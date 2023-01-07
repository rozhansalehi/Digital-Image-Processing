import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

################## Part B ######################
def transform(img1 , bd):
    img1_copy = img1.copy() 
    img1_copy = img1_copy.astype(np.float32) # change image type to float in order to perform intensity operation on it
    a = 1/(2**bd) # it has been calculated in the workreprt: a=1/L
    img2 = a * (img1_copy**2 +img1_copy) #transform function
    img2 = np.round( img2) # quantizing
    img2=img2.astype(img1.dtype) # input dtype = output dtype
    return img2

################## Part C ######################
kidney = cv.imread('kidney.tif' , 0)
plt.figure(1)
plt.subplot(2,2,1), plt.imshow(kidney,'gray'), plt.title('main image',fontsize=10),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(transform(kidney,8),'gray'), plt.title('Transformed image',fontsize=10),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.hist(kidney.ravel(), bins=255//4), plt.title('main image histogram',fontsize=10)
plt.subplot(2,2,4), plt.hist(transform(kidney,8).ravel(), bins=255//4), plt.title('Transformed image histogram',fontsize=10)
plt.savefig('kidney')

chest = cv.imread('chest.tif' , cv.IMREAD_ANYDEPTH)
plt.figure(2)
plt.subplot(2,2,1), plt.imshow(chest , 'gray'), plt.title('main image',fontsize=10),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(transform(chest,16), 'gray'), plt.title('Transformed image',fontsize=10),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.hist(chest.ravel(), bins=255//4), plt.title('main image histogram',fontsize=10)
plt.subplot(2,2,4), plt.hist(transform(chest,8).ravel(), bins=(2**16-1)//1024), plt.title('Transformed image histogram',fontsize=10)
plt.savefig('chest')
plt.show()

################## Part D ######################
plt.figure(3)
r = np.arange(0,2**16-1)  # initial intensities
x = r.copy() # Get copy of r in order to not change in the post processes
identity = x
power = transform(x,16)
plt.plot(r , identity , color='b' , linestyle='--', label='identity transform')
plt.plot(r , power , color='k', label='power law(Gamma) transform')
plt.legend()
plt.savefig('transform function')
plt.show()