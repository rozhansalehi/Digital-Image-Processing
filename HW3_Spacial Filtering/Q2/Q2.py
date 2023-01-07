import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#################### Part 1 & 2 #####################
def filtering( img1 , filter_name , n ): # n: size of the kernel
    N,M = img1.shape
    img2 = np.zeros_like(img1 , dtype = 'float32') 

    # Padding
    pad_size = (n-1)//2
    img1_pad = np.pad(img1 , pad_size , mode='reflect')
     
    for y in range(pad_size , M+pad_size):
        for x in range(pad_size , N+pad_size):
            if filter_name == 'minimum':
                img2[x-pad_size , y-pad_size] = np.min(img1_pad[x-pad_size : x+pad_size+1, y-pad_size : y+pad_size+1 ])     
            
            elif filter_name == 'median':
                img2[x-pad_size , y-pad_size] = np.median(img1_pad[x-pad_size : x+pad_size+1, y-pad_size : y+pad_size+1 ])
                
            elif filter_name == 'averaging':
                w = np.full( (n,n), 1/(n*n) )
                img2[x-pad_size,y-pad_size] = np.sum( w*img1_pad[x-pad_size : x+pad_size+1, y-pad_size : y+pad_size+1 ] )#Convolving  
            
            # n*n sobel filter: Sobel_x[i,j] = i / (i*i + j*j), Sobel_y[i,j] = j / (i*i + j*j)
            elif filter_name == 'sobel_y':  
                w = np.zeros( (n,n) )      
                for i in range( -1*pad_size , pad_size+1):
                    for j in range( -1*pad_size , pad_size+1):
                        if j!=0 :
                            w[i+pad_size,j+pad_size] = j / (i*i + j*j)    
                img2[x-pad_size,y-pad_size] = np.sum( w*img1_pad[x-pad_size : x+pad_size+1, y-pad_size : y+pad_size+1 ] )#Convolving              
            
            elif filter_name == 'laplacian':
                w = np.array([[ 0 ,-1 , 0 ],
                                [-1 , 4 , -1],
                                [ 0 ,-1 , 0 ]])
                img2[x-pad_size,y-pad_size] = np.sum( w*img1_pad[x-pad_size : x+pad_size+1, y-pad_size : y+pad_size+1 ] )#Convolving  
            
            elif filter_name == 's':
                w = np.array([[-1 ,-1 , 0],
                                [-1 , 0 , 1],
                                [ 0 , 1 , 1]])

                img2[x-pad_size,y-pad_size] = np.sum( w*img1_pad[x-pad_size : x+pad_size+1, y-pad_size : y+pad_size+1 ] )#Convolving  

    return img2 

#################### Part 3 #####################
main_img = cv.imread('MRI.png', 0)                
average1 = filtering(main_img , 'averaging', 3)
average2 = filtering(main_img , 'averaging', 7)
minimum1 = filtering(main_img , 'minimum', 3)
minimum2 = filtering(main_img , 'minimum', 7)
median1 = filtering(main_img , 'median', 3)
median2 = filtering(main_img , 'median', 7)
sobel_y = filtering(main_img , 'sobel_y', 3)
laplacian = filtering(main_img , 'laplacian', 3)
s = filtering(main_img , 's', 3)

plt.figure('filtering')
plt.subplot(3,4,1), plt.imshow(average1 , cmap='gray'), plt.title('average, 3*3 kernel'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,5), plt.imshow(average2 , cmap='gray'), plt.title('average, 7*7 kernel'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,2), plt.imshow(minimum1 , cmap='gray'), plt.title('minimum, 3*3 kernel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,4,6), plt.imshow(minimum2 , cmap='gray'), plt.title('minimum, 7*7 kernel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,4,3), plt.imshow(median1 , cmap='gray'), plt.title('median, 3*3 kernel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,4,7), plt.imshow(median2 , cmap='gray'), plt.title('median, 7*7 kernel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,4,4), plt.imshow(sobel_y , cmap='gray'), plt.title('sobel_y, 3*3 kernel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,4,8), plt.imshow(laplacian , cmap='gray'), plt.title('laplacian, 3*3 kernel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,4,12), plt.imshow(s , cmap='gray') , plt.title('s, 3*3 kernel'),plt.xticks([]), plt.yticks([])
plt.suptitle('filtering')

plt.show()
