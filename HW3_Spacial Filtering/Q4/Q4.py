import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

######################## Part a #########################
def bitplane_slice(image , n):
    plane = np.full(image.shape , 2**n , np.uint8)
    bitplane_n = cv.bitwise_and(image , plane) # by using bitwise_and we can slice nth bit_palne
    return bitplane_n

######################## Part b #########################
PCB = cv.imread('PCB.tif' , 0)

fig,a = plt.subplots(2,4)
PCB_bit = np.zeros((8,PCB.shape[0], PCB.shape[1]))
for n in range(0,8):
    PCB_bit[n,...] =  bitplane_slice(PCB , n)
    a[n//4,n%4].imshow(PCB_bit[n], cmap='gray')
    a[n//4,n%4].set_title('bit number:'+str(n))
    a[n//4,n%4].set_xticks([])
    a[n//4,n%4].set_yticks([])    
plt.suptitle('Bit plane slicing')
plt.savefig('Bit Plane Slicing.tif')    
plt.show()
######################## Part c #########################
NASA_A = cv.imread('NASA-A.tif' , 0)
NASA_B = cv.imread('NASA-B.tif' , 0)
NASA_C = cv.imread('NASA-C.tif' , 0)

k = [16,32,64,128] # Coefficients of each bitplane
sub_AB= np.zeros(NASA_A.shape)
sub_BC = np.zeros(NASA_A.shape)
for n in range(4,8):
    sub_AB += k[n-4]/256 * cv.bitwise_xor(bitplane_slice(NASA_A,n), bitplane_slice(NASA_B,n) )
    sub_BC += k[n-4]/256* cv.bitwise_xor(bitplane_slice(NASA_B,n), bitplane_slice(NASA_C,n) )

plt.figure(2)
plt.subplot(121), plt.imshow(sub_AB,cmap='gray', vmin=0,vmax=255), plt.title('subtraction of image A & B'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(sub_BC,cmap='gray', vmin=0,vmax=255), plt.title('subtraction of image B & C'), plt.xticks([]), plt.yticks([])
plt.suptitle('motion detection')
plt.savefig('motion detection.tif')
plt.show()
