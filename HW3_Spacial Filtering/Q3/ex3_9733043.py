
import matplotlib.pyplot as plt 
import numpy as np
import cv2 as cv

retina = cv.imread('retina.jpg',0)
median = cv.medianBlur(retina, 3)
averaging = cv.blur(retina, (3,3))

plt.figure(1)
plt.subplot(3,1,1)
plt.imshow(retina, cmap='gray')

plt.subplot(312)
plt.imshow(median, cmap='gray')

plt.subplot(313)
plt.imshow(averaging, cmap='gray')
plt.show()

def power(img,landa):
    img_clip = np.clip(img,0,255)
    c = 255**(1-landa)
    s = c*img_clip**landa
    s= s.astype('uint8')
    

    return s

plt.figure(2)
image_c = power(median,2/3)
plt.imshow(image_c, cmap='gray',vmin = 0,vmax=255)

plt.show()

def laplacian(img):
    img_pad = np.pad(img, 1, mode='reflect')
    a, b = img.shape 
    kernel = np.array([
            [-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1]
        ])
    new_img = np.zeros((a,b))

    for i in range(1, a+1):
        for j in range(1, b+1):
            new_img[i-1, j-1]  = np.sum(img_pad[i-1:i+2,j-1:j+2]*kernel)
    return new_img

mask = laplacian(image_c) 
plt.figure(3)
power_mask = power(mask,1/3)
plt.imshow(power_mask, cmap='gray')

plt.show()
c = np.arange(-2,2,0.01)
framesize = image_c.shape
video = cv.VideoWriter('video.avi',cv.VideoWriter_fourcc(*'DIVX'),20,framesize,0)
mask = mask.astype('float32')
image_c = image_c.astype('float32')

for i in range(c.size):
    video_frame = image_c +c[i]*mask
    video_frame = np.clip(video_frame,0,255)
    video_frame = video_frame.astype('uint8')
    video.write(video_frame)

video.release()


