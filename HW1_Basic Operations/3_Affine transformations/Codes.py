import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

######################### Part A ######################
# Reading the image in gray
AUT_DIP=cv.imread('AUT-DIP.png', 0)

# Cropping it into 6 pieces
height1 = int ( AUT_DIP.shape[0]/2 )
height2 = int ( AUT_DIP.shape[0] )

width1 = int ( AUT_DIP.shape[1]/ 3)
width2 = int ( AUT_DIP.shape[1]*2/3 )
width3 = int ( AUT_DIP.shape[1] )

A = AUT_DIP [ 0:height1, 0:width1 ]
U = AUT_DIP [ 0:height1, width1+1:width2 ]
T = AUT_DIP [ 0:height1, width2+1:width3 ]
D = AUT_DIP [ height1+1:height2, 0:width1 ]
I = AUT_DIP [ height1+1:height2, width1+1:width2 ]
P = AUT_DIP [ height1+1:height2, width2+1:width3 ]

plt.figure(1)
plt.subplot(231), plt.imshow(A, cmap='gray', vmin=0, vmax=255), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(U,'gray', vmin=0, vmax=255), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(T, 'gray', vmin=0, vmax=255), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(D, 'gray', vmin=0, vmax=255), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(I, 'gray', vmin=0, vmax=255), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(P, 'gray', vmin=0, vmax=255), plt.xticks([]), plt.yticks([])
plt.suptitle('AUT_DIP Cropped')
plt.savefig('AUT-DIP Cropped.png')
plt.show()

######################### Part B #######################
# Scaling 2X
A_scaling = cv.resize(A,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

# Cropping it 
Ay_center = A_scaling.shape[0]/2
Ax_center = A_scaling.shape[1]/2
cropped_height1 = int( Ay_center - A.shape[0]/2 )
cropped_height2 = int( Ay_center + A.shape[0]/2 )
cropped_width1 = int( Ax_center - A.shape[1]/2 )
cropped_width2 = int( Ax_center + A.shape[1]/2 )
A_cropping=A_scaling [ cropped_height1+1 :cropped_height2, cropped_width1:cropped_width2 ]

plt.figure(2)
plt.subplot(131), plt.imshow( A,'gray' ), plt.title( 'A' )
plt.subplot(132), plt.imshow( A_scaling,'gray' ), plt.title( 'scaled' )
plt.subplot(133), plt.imshow( A_cropping,'gray' ), plt.title( 'cropped' )
print('A dim:',A.shape, 'A_scaling dim:', A_scaling.shape, 'A_cropping dim:',A_cropping.shape)
plt.show()

####################### Part C ###########################
# Horizontal shearing
shearing_mat = np.float32( [ [1 ,0.2, 0],
                             [0 , 1 , 0] ])
sheared_U = cv.warpAffine( U, shearing_mat, U.shape )
plt.figure(3)
plt.imshow(sheared_U,'gray')
plt.show()

###################### Part D ###########################
# Translating
translating_mat = np.float32([ [1 , 0 , -80],
                               [0 , 1 , 100] ])
translated_T=cv.warpAffine(T, translating_mat, T.shape)
plt.figure(4)
plt.imshow(translated_T,'gray')
plt.show()  
  
###################### Part E ##########################
# Forward rotating
forward_rotated_D = np.zeros(D.shape)
angle1_DEG=25
angle1_RAD=angle1_DEG*(np.pi/180)

for i in range( D.shape[0] ):
    for j in range( D.shape[1] ):
        forward_x = int( i*np.cos(angle1_RAD) - j*np.sin(angle1_RAD) )
        forward_y = int( i*np.sin(angle1_RAD) + j*np.cos(angle1_RAD) )
        if (forward_x < D.shape[0]) and (forward_y < D.shape[1]):
            forward_rotated_D[i, j] = D[forward_x, forward_y]

forward_rotated_D = forward_rotated_D.astype('uint8')
forward_rotated_D = cv.resize(forward_rotated_D,(500,499),fx=1, fy=1)
plt.figure(5)
plt.imshow(forward_rotated_D,'gray')
plt.savefig('rotated D.png')
plt.show()

######################### Part F ########################
# Inverse Rotating
inverse_rotated_I=np.zeros(I.shape)
angle2_DEG=-25
angle2_RAD=angle2_DEG*(np.pi/180)

for i in range( I.shape[0] ):
    for j in range( I.shape[1] ):
        inverse_x = int( i*np.cos(angle2_RAD) + j*np.sin(angle2_RAD) )
        inverse_y = int( -1 *i*np.sin(angle2_RAD) + j*np.cos(angle2_RAD) )
        if (inverse_x < I.shape[0]) and (inverse_y < I.shape[1]):
            inverse_rotated_I[i, j] = I[inverse_x, inverse_y]

inverse_rotated_I = inverse_rotated_I.astype(np.uint8)
inverse_rotated_I = cv.resize(inverse_rotated_I,(500,499),fx=1, fy=1)
plt.figure(6)
plt.imshow( inverse_rotated_I,'gray')
plt.show()

######################### Part G #########################
# Rotating
rotated_mat=cv.getRotationMatrix2D((P.shape[0]/2,P.shape[1]/2),45,1)
Rotated_P=cv.warpAffine(P,rotated_mat,(P.shape[0],P.shape[1]))
Rotated_P=cv.resize(Rotated_P,(500,499),fx=1, fy=1)
plt.figure(7)
plt.imshow(Rotated_P,'gray')
plt.show()
print(A_cropping.shape , sheared_U.shape, translated_T.shape)
print(forward_rotated_D.shape,inverse_rotated_I.shape,Rotated_P.shape)
print(forward_rotated_D.dtype,inverse_rotated_I.dtype,Rotated_P.dtype)

######################### Part H ##########################
# Concatenating
Hconcat1=cv.hconcat( [A_cropping , sheared_U , translated_T] )
Hconcat2=cv.hconcat([forward_rotated_D, inverse_rotated_I , Rotated_P])
plt.figure(8)
concatenated_image=cv.vconcat([Hconcat1,Hconcat2])
plt.imshow(concatenated_image,'gray')
plt.savefig('concatenated_image')
plt.show()

                 