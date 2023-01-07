import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

################## Part B ######################
def transform2(img1 , A , B):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i,j]<A or img1[i,j]>B:
                    img1[i,j]=0  # intensities which are out of the [A,B] range, are changed to zero
    return img1

################## Part C ######################
# Reading the image in gray
HeadCT = cv.imread('HeadCT.tif' , 0)

# get copy of main image in order to not change in the process
img_copy = HeadCT.copy()

# Estimate the intensities if bone and baack ground 
bone_intensity = img_copy[460,300]
background_intensity = img_copy[50,10]
print('bone intensity=',bone_intensity,'\n','background intensity=', background_intensity)

New_img = transform2(img_copy,background_intensity+10, bone_intensity-10 ) # +-10 as confidence range
plt.figure(1)
plt.subplot(121), plt.imshow(HeadCT, 'gray'), plt.title('Main Image')
plt.subplot(122), plt.imshow(New_img, 'gray',vmin=0,vmax=HeadCT.max()), plt.title('New Image')
plt.savefig('main & new images')
plt.show()

################## Part D ######################

r = np.arange(0,2**8-1) # initial intensities
z = r.copy() # Get copy of r in order to not change in the post processes
y = z[np.newaxis,: ] # add new dimension to initial intensities in order to be compatible with input of transform2 function
s=transform2(y , background_intensity+10 , bone_intensity-10).ravel() # output of transform2 function is 2D, thus  for plotting it we change it to 1D 

plt.figure(2)
plt.plot(r,s,color='g')
plt.savefig('slicing transform')
plt.show()