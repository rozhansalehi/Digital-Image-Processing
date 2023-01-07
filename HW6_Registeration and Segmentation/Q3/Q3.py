import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
######################## Part C ######################
img = cv.imread('CBC.jpg', 0) #read image in "gray"
edge = cv.Canny(img,20,40) # edge detection by applying Canny filter

plt.figure(1)
plt.imshow(edge, cmap='gray'), plt.title('detected edges'), plt.axis(False)
plt.savefig('detected edges.png')
plt.show()
######################## Part D ######################
# input image in cv.HoughCircles must be gray
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp = 0.75, 
                        minDist=130, param1=40, param2=20, minRadius=70, maxRadius=160)

j = circles.shape
print('number of detected circles:',j[1])

# to draw colored circle on main image, it should be colored too                        
img_colored = cv.cvtColor(img,cv.COLOR_GRAY2BGR) 

if circles is not None:
    # center & radius of circles must be integer in "cv.circle" function and they need 16 bit memory  
    circles = np.uint16(np.around(circles)) 
                                            
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2] 
        cv.circle(img_colored, center, radius, (0, 255, 0), 5) # draw the outer circle  
        cv.circle(img_colored, center, 1, (0, 0, 255), 5) # draw the center of the circle

cv.imshow("detected circles", img_colored)
if cv.waitKey(0) == ord("s"):
    cv.imwrite('detected circles.png',img_colored)
cv.destroyAllWindows
      