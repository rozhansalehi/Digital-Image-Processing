import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
####################### Part a ######################
main_img = cv.imread('retina.jpg' , 0)
N , M = main_img.shape

# Padding: pad_size=1
img_pad = np.pad(main_img, 1 , mode='reflect') # Copying main_img in order not to change

avg_img =  np.zeros((N , M))
median_img = np.zeros((N , M))
avg_median =  np.zeros((N , M))

# Applying average andmedian filter seperately
avg_w = np.full((3,3) , 1/9) #Average kernel
for y in range(1 , M+1):
    for x in range(1 , N+1):
        median_img[x-1 , y-1] = np.median(img_pad[x-1 : x+2, y-1 : y+2 ])
        avg_img[x-1,y-1] = np.sum( avg_w*img_pad[x-1 : x+2, y-1 : y+2 ])#Convolving  

# Applying average filter to median_img
median_img_pad = np.pad(median_img , 1 , mode='reflect') 
for y in range(1 , M+1):
    for x in range(1 , N+1):
        avg_median[x-1,y-1] = np.sum(avg_w *median_img_pad[x-1 : x+2, y-1 : y+2])   #Convolving

plt.figure('median image')
plt.subplot(131), plt.imshow(median_img , cmap='gray', vmin=0,vmax=255), plt.title('median img'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(avg_img , cmap='gray', vmin=0,vmax=255), plt.title('average img'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(avg_median , cmap='gray', vmin=0,vmax=255), plt.title('both average & median img',fontsize = 10), plt.xticks([]), plt.yticks([])
plt.savefig('media_average_both.png')
plt.show()

####################### Part b ######################
def power(img1 , gamma):
    a = 255**(1-gamma)
    img1_clip = np.clip(img1 , 0,  255 ) # clipping the image in order to not be negative and not to overflow
    img2 = a * img1_clip ** gamma
    img2 = img2.astype('uint8') # Intensity of output image is 8 bit
    return img2

####################### Part c ######################
img_c = power(median_img , 2/3)

plt.figure('power transformation')
plt.imshow(img_c , cmap='gray', vmin=0,vmax=255), plt.title('img_c: power_transformed'), plt.xticks([]), plt.yticks([])
plt.savefig('img_c.png')
plt.show()

####################### Part d ######################
laplacian_img = np.zeros((N,M),dtype='float32')
img_c_pad = np.pad(img_c , 1 , mode='reflect')
laplacian_w = np.array([[-1 , -1 ,-1],
                        [-1 , 8 , -1],
                        [-1 , -1 , -1]])
for y in range(1,M+1):
    for x in range(1,N+1):                
        laplacian_img[x-1,y-1] = np.sum(laplacian_w * img_c_pad[x-1:x+2 , y-1:y+2])
plt.figure('mask(laplacian)')
plt.imshow(laplacian_img, cmap='gray',vmin=0,vmax=255), plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.savefig('mask(laplacian).png')

mask_power_clip = np.clip( power(laplacian_img, 1/3) , 0 , 255)
plt.figure('mask_power')
plt.imshow(mask_power_clip , cmap='gray', vmin=0,vmax=255), plt.title('mask with better visualization'), plt.xticks([]), plt.yticks([])
plt.savefig('mask with better visualization.png')
plt.show()

####################### Part e ######################
# dtype = float
c = np.arange(-2 , 2.01 , 0.01)
A,B = img_c.shape
out1 = cv.VideoWriter('output1_video.mp4',cv.VideoWriter_fourcc(*'mp4v') , 20 ,(A,B) , 0)
img_c_float = img_c.astype('float32')

for i in range(c.size):
    frame = img_c_float + c[i] * laplacian_img
    frame = np.clip(frame , 0 , 255)
    frame = frame.astype('uint8')
    out1.write(frame)
out1.release()

# dtype = uint8
c = np.arange(-2 , 2.01 , 0.01)
A,B = img_c.shape
out2 = cv.VideoWriter('output2_video.mp4',cv.VideoWriter_fourcc(*'mp4v') , 20 ,(A,B) , 0)
img_c_uint8 = img_c.astype('uint8')
laplacian_img_uint8 = laplacian_img.astype('uint8')
for i in range(c.size):
    frame = img_c_uint8 + c[i] * laplacian_img_uint8
    frame = np.clip(frame , 0 , 255)
    frame = frame.astype('uint8')
    out2.write(frame)
out2.release()