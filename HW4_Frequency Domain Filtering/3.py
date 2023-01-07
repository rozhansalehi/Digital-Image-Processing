import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pydicom
dataset1 = pydicom.dcmread('Thoracic CT 1.dcm') #reading dicom file
img1 = dataset1.pixel_array # access pixels
dataset2 = pydicom.dcmread('Thoracic CT 2.dcm')
img2 = dataset2.pixel_array

bit_depth = 12 #?
L_1 = 2**bit_depth - 1


mask = np.fromfunction(lambda i, j: (-1)**(i+j), img1.shape, dtype=int) # mask for shifting dc to center in frequency domain

img1_prime = img1 * mask
img2_prime = img2 * mask

fig, axs = plt.subplots(2, 2)
plt.suptitle('Choose the right option')

axs[0, 0].set_title('boundary of img1 is correct')
axs[0, 0].imshow(img1, cmap='gray', vmin=0, vmax=L_1)
axs[0, 0].axis(False)

axs[1, 0].set_title('boundary of img1 is stretched')
axs[1, 0].imshow(img1, cmap='gray', vmin=-L_1, vmax=L_1)
axs[1, 0].axis(False)


axs[0, 1].set_title('boundary of img1_prime is clipped')
axs[0, 1].imshow(img1_prime, cmap='gray', vmin=0, vmax=L_1)
axs[0, 1].axis(False)

axs[1, 1].set_title('boundary of img2_prime is correct')
axs[1, 1].imshow(img2_prime, cmap='gray', vmin=-L_1, vmax=L_1)
axs[1, 1].axis(False)

plt.tight_layout()

IMG1 = np.fft.fft2(img1_prime) # compute 2D DFT
IMG1_abs = np.abs(IMG1) # spectrum
IMG1_abs_log = np.log(IMG1_abs+1) # better visualization of low intensities
IMG1_angle = np.angle(IMG1) # phase

IMG2 = np.fft.fft2(img2_prime)
IMG2_abs = np.abs(IMG2)
IMG2_abs_log = np.log(IMG2_abs+1)
IMG2_angle = np.angle(IMG2)


maxI = img1.size*L_1 # maximum intensity ofspectrum
maxI_log = np.log(maxI+1) # better visualization of low intensities


plt.figure() 
plt.suptitle('Images in frequency domain')

plt.subplot(231)
plt.title(r'${\left| {{IMG1}} \right|}$')
plt.imshow(IMG1_abs, cmap='gray', vmin=0, vmax=maxI)
plt.axis(False)

plt.subplot(232)
plt.title(r'$\log \left( {\left| {{IMG1}} \right| + 1} \right)$')
plt.imshow(IMG1_abs_log, cmap='gray', vmin=0, vmax=maxI_log)
plt.axis(False)


plt.subplot(233)
plt.title(r'$\angle$IMG1: $\left( { - \pi ,\pi } \right]$')
plt.imshow(IMG1_angle, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.axis(False)

plt.subplot(234)
plt.title(r'${\left| {{IMG2}} \right|}$')
plt.imshow(IMG2_abs, cmap='gray', vmin=0, vmax=maxI)
plt.axis(False)

plt.subplot(235)
plt.title(r'$\log \left( {\left| {{IMG2}} \right| + 1} \right)$')
plt.imshow(IMG2_abs_log, cmap='gray', vmin=0, vmax=maxI_log)
plt.axis(False)

plt.subplot(236)
plt.title(r'$\angle$IMG2: $\left( { - \pi ,\pi } \right]$')
plt.imshow(IMG2_angle, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.axis(False)

IMG1_c =  np.conj(IMG1) # Re{IMG1} - j Im{IMG1} --> 90 degree rotation
IMG2_c = -np.conj(IMG2) # -Re{IMG1} + j Im{IMG1} --> 90 degree rotation

img1_c_shifted = np.fft.ifft2(IMG1_c).real
img2_c_shifted = np.fft.ifft2(IMG2_c).real


img1_c = np.clip(img1_c_shifted * mask, 0, L_1) 
img2_c = np.clip(img2_c_shifted * mask, -L_1, 0) + L_1

plt.figure()

plt.subplot(221)
plt.title('img1')
plt.imshow(img1, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.subplot(222)
plt.title('img2')
plt.imshow(img2, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.subplot(223)
plt.title('what happend?') 
plt.imshow(img1_c, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.subplot(224)
plt.title('what happend?')
plt.imshow(img2_c, cmap='gray', vmin=0, vmax=L_1)
plt.axis(False)

plt.show()