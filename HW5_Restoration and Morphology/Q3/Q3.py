import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
################## Part A ##################
noisy_img = cv.imread('noisy_image.png' , 0 )
mask = np.fromfunction(lambda i, j: (-1)**(i+j), noisy_img.shape, dtype=int) # mask for shifting dc to center in frequency domain
img_shifted = noisy_img * mask

# DFT
IMG = np.fft.fft2(img_shifted) # compute 2D DFT
IMG_abs = np.abs(IMG) # spectrum
IMG_abs_log = np.log(IMG_abs+1) # better visualization of low intensities

maxI = 255 * noisy_img.size # maximum intensity of spectrum
maxI_log = np.log(maxI+1) # better visualization of low intensities

# Filters
notch_pass = np.zeros(noisy_img.shape)
x, y = np.indices(notch_pass.shape)
r, c = notch_pass.shape
d0 = np.sqrt(r**2 + c**2)
dis = (x-r//2)**2 + (y-c//2)**2
notch_pass[((dis < 150.5**2) ^ (dis < 149.5**2)) | ((y<(c//2 -0.5)) ^ (y<(c//2 +0.5))) | ((x<(r//2 -0.5)) ^ (x<(r//2 +0.5)))] = 1
notch_reject = 1 - notch_pass

# Apllying filter
FILTERED_IMG = IMG * notch_reject
FILTERED_IMG_abs =np.abs( IMG * notch_reject)
FILTERED_IMG_log = np.log(FILTERED_IMG_abs + 1)

# IDFT
filtered_img = np.fft.ifft2(FILTERED_IMG).real
new = np.clip(filtered_img * mask,0 , 255)

plt.figure(1)
plt.suptitle('Images in frequency domain')
plt.subplot(121)
plt.title(r'${\left| {{IMG1}} \right|}$')
plt.imshow(IMG_abs, cmap='gray', vmin=0, vmax=maxI)
plt.axis(False)
plt.subplot(122)
plt.title(r'$\log \left( {\left| {{IMG1}} \right| + 1} \right)$')
plt.imshow(IMG_abs_log, cmap='gray', vmin=0, vmax=maxI_log)
plt.axis(False)
plt.savefig('Frequency Domain.png')

plt.figure(2)
plt.suptitle('filter & filtered image in frequency domain')
plt.subplot(121)
plt.title('notch reject filter')
plt.imshow(notch_reject, cmap='gray')
plt.axis(False)
plt.subplot(122)
plt.title('filtered image in freq domain')
plt.imshow(FILTERED_IMG_log, cmap='gray', vmin=0, vmax=maxI_log)
plt.axis(False)
plt.savefig('filter & filtered image in frequency domain.png')

plt.figure(3)
plt.title('filtered image in position domain')
plt.imshow(new, cmap='gray',vmin=0, vmax=255)
plt.axis(False)
plt.savefig('filtered image in position domain.png')
plt.show()