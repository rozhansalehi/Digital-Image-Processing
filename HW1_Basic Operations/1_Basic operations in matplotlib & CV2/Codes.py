import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

######################## Part A #######################
# Reading the image in gray:  
img1 = cv.imread('Head-MRI.tif', 0)

# Defining types of image1 & image2:
img1 = img1.astype('uint8')
img2 = img1.astype('float')

plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.title('uint8 image'),
plt.xticks([])
plt.yticks([])

plt.subplot(2,1,2)
plt.imshow(img2/255, cmap='gray', vmin=0, vmax=1)
plt.title('float image')
plt.xticks([])
plt.yticks([])
plt.savefig('uint8 and float images.png')
plt.show()


print('image1 type:',img1.dtype,'\n','image2 type:', img2.dtype)

########################## Part B ########################
row150=img1[150,:]
row180=img1[180,:]

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(row150,color='b',label='row 150')
plt.plot(row180,color='r',label='row 180')
plt.title('Intensity of rows 150 & 180 of Head-MRI image')
plt.legend()

######################## Part C #########################
# Defining a 2*img1.shape([1]) array to put row150 and row 180 in it
img = np.zeros((2,img1.shape[1]))
img[0,:]=img1[150,:]
img[1,:]=img1[180,:]
plt.subplot(2,1,2)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])
plt.title('Rows 150 & 180 of the image')
plt.savefig('Rows 150 & 180 of the image')
plt.show()



