import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
avg =  np.zeros((5, 5))
w = np.array([[-1 , -1 , 0],
                [-1 , 0, 1],
                [0 , 1 , 1],])
a=np.array([[0,0,10,10,0],
            [0,10,20,20,10],
            [0,10,20,20,10],
            [0,0,10,10,0],
            [0,0,0,0,0],])
a_pad = np.pad(a , 1 , mode='constant',constant_values=0) 
for y in range(1 , 5+1):
    for x in range(1 , 5+1):
        avg[x-1,y-1] = np.sum(w *a_pad[x-1 : x+2, y-1 : y+2])            
print(a_pad,'\n',avg)