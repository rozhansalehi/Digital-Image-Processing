import numpy as np
import cv2 as cv

###################### Part A ######################
# Reading the Image in gray:
MRI_Head = cv.imread('MRI-Head.png',0)

###################### Part B ######################
# Capturing the video and the first frame in gray:
cap = cv.VideoCapture('MRI.avi')
cap.set(cv.CAP_PROP_POS_FRAMES,1)
ret, frame1 = cap.read()
frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

cv.imshow('frame1', frame1)
k = cv.waitKey(0)
if k == ord("o"):
   cv.imwrite('frame1.png', frame1)

# Finding the noise of the frame1:
MRI_Head = np.int32(MRI_Head) # Changing the images type into 'int32' in order to subtract them
frame1_gray = np.int32(frame1_gray)
noise1 = MRI_Head - frame1_gray

# Calculating mean & standard deviation of noise1
mean1 = noise1.mean()
std1 = noise1.std()
print('mean of noise of the frame1:',mean1,'\n','StD of noise of the frame1:',std1)

noise1 = noise1.astype('uint8')# Changing the noise1 type into 'uint8' in order to show that with opencv
cv.imshow('noise1',noise1)
k = cv.waitKey(0)
if k == ord("o"):
   cv.imwrite('noise1.png', noise1)

# calculate average of all frames:
counter = 1
frames_sum = np.zeros(frame1.shape)
while cap.isOpened():
   ret, frames = cap.read()
   if ret:
      frames_sum = frames + frames_sum
      counter+=1 
   else:
      break

cap.release()
cv.destroyAllWindows()

frames_avg = frames_sum/counter
frames_avg = frames_avg.astype('uint8')# Changing the frames_avg type into 'uint8' in order to show that with opencv
cv.imshow('frames_avg',frames_avg)
k = cv.waitKey(0)
if k == ord("o"):
   cv.imwrite('frames_avg.png', frames_avg)

########################### Part C #######################
# reducing frames_avg dimension to 2D
frames_avg_gray = cv.cvtColor(frames_avg, cv.COLOR_BGR2GRAY)

frames_avg_gray = frames_avg_gray.astype('int32') # Changing the images type into 'int32' in order to subtract them
MRI_Head = MRI_Head.astype('int32')
noise2 = MRI_Head - frames_avg_gray

# Calculating mean & standard deviation of noise2
mean2 = noise2.mean()
std2 = noise2.std()
print('mean of noise of frames_avg:',mean2,'\n','StD of noise of frames_avg:',std2)
print('number of frames=',counter)
noise2 = noise2.astype('uint8')# Changing the noise2 type into 'uint8' in order to show that with opencv
cv.imshow('noise2',noise2)
k=cv.waitKey(0)
if k==ord("o"):
   cv.imwrite('noise 2.png', noise2)

########################## Part D ########################
# Mask
mask = cv.imread('mask.png', 0)
MRI_Head = MRI_Head.astype('uint8')# Changing the MRI-Head type into 'uint8' in order to use cv.subtract() 

m1 = cv.subtract(mask, MRI_Head)
m2 = cv.subtract(mask, m1)

cv.imshow('covered image',m2)
k = cv.waitKey(0)
if k == ord("o"):
   cv.imwrite('covered image.png', m2)
