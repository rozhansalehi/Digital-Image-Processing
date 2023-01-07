import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
############### Using 3 points by writing them in the code ################
img1 = cv.imread('MRI.jpg', 0)
img2 = cv.imread('MRI2.jpg', 0)

pts1 = np.float32([[162, 35.6], [191, 101], [128.5, 93.6]])
pts2 = np.float32([[23, 104], [166, 128], [84, 219.6]])

affine1 = cv.getAffineTransform(pts1, pts2)
print('affine matrix (first method):\n', affine1)
img_affined_1 = cv.warpAffine(img1, affine1, img2.shape)

plt.figure(1)
plt.imshow(img_affined_1,'gray'), plt.title('affining by first method'), plt.axis(False)
plt.savefig('first method.png')

################ Using more than 3 points by by getting them from user #####################
def get_points(path):
    pt = []
    def draw_circles(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDBLCLK :
            cv.circle(img, (x,y), 2, (0,0,255),-1)
            pt.append((x,y))
            
    # get points from input image
    img = cv.imread(path)
    cv.namedWindow('window')
    cv.setMouseCallback('window', draw_circles)
    while 1:
        cv.imshow('window', img)
        if cv.waitKey(20) & 0xFF == 27: #ASCII code of "esc" button
            cv.imwrite('get points.png', img)
            break
    points = pt.copy()
    cv.destroyAllWindows()
    return points 
    
points_1 = get_points('MRI.jpg')
print('points list 1:', points_1)
points_2 = get_points('MRI2.jpg')
print('points list 2:',points_2)

# finding beta matrices
j = np.array(points_2)
beta_x = [i for i in j[:,0]]
beta_y = [i for i in j[:,1]]

# finding M matrix
k = np.array(points_1)
k_T = np.transpose(k)
r,c = k.shape
one_vector = np.ones((r,1))
M = np.hstack((k,one_vector))

# finding alpha matrix (affine matrix)
M_T = np.transpose(M)
square_mat = np.matmul(M_T, M)
square_mat_inv = np.linalg.inv(square_mat)

img1_T_beta_x = np.matmul(M_T,beta_x)
alpha_1 = np.matmul(square_mat_inv,img1_T_beta_x)

img1_T_beta_y = np.matmul(M_T,beta_y)
alpha_2 = np.matmul(square_mat_inv,img1_T_beta_y)

affine2 = np.vstack((alpha_1, alpha_2))
print('affine matrix (second method):\n', affine2)

img_affined_2 = cv.warpAffine(img1, affine2, img2.shape)
plt.figure(2)
plt.imshow(img_affined_2,'gray'), plt.title('affining by second method'), plt.axis(False)
plt.savefig('second method.png')
plt.show()