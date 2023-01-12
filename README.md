# Digital-Image-Processing
Various operations for enhancing the images

## HW1: Basic Operations 
### 1. Reading and Showing images, understanding data types
- Reading the images in grayscale: cv2.IMREAD_GRAYSCALE
- Changing data type: .astype
- Accessing the pixels through indexing

### 2. Averaging video sequences
- Reading a video: cv2.VideoCapture()
- Calculating the average and variance of video sequences
- Masking an image

### 3. Affine transformations
- Defining a function to return an n-bit image
- Using thresholding functions in cv2: cv.THRESH_BINARY, cv.THRESH_BINARY_INV, cv.THRESH_TRUNC, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV
- Thresholding for segmentation

### 4. Changing color space and thresholding
- Cropping an image by indexing
- Scale transform: cv.resize()
- Understanding interpolation flags in cv.resize(): cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_AREA, cv.INTER_CUBIC
- Using cv.warpAffine() for affine transformations: shear, translation, rotation

## HW2: Intensity-based Operations
### 1. Histogram Equalization
- Plotting histogram
- Applying power law transform
- Comparing the effect of exponential and logarithmic functions on contrast

### 2. Transform function for increasing contrast

## HW3: Spatial Filtering
### 1. Edge-detecting filters
- Padding the input image
- Defining mean, median, Sobel, and laplacian kernels

### 2. Denoising filters
- Local-based filtering
- Intensity-based filtering
- Combining both methods

### 3. Bit-plane slicing & motion detection
- Bit-wise operations (low significant bits contain details while high significant bits contain the generality of the image)
- Using bit-wise XOR operation to detect the difference between some frames which leads to motion detection

## HW4: Frequency Domain Filtering
- Preserving the intensity boundaries during filtering
- Stretching and clipping
- Spatial information in phase image of Fourier transform

## HW5: Restoration and Morphology
### 1. Removing spatial-pattern noise
- Identifying noise in the Fourier transformation of the image
- Applying a notch filter
- Getting inverse Fourier transform

### 2. Morphology operations for denoising
- Binarizing the image with a specific threshold
- Opening operation (erosion -> dilation -> erosion)
- Closing operation (dilation -> erosion -> dilation)

### 3. Hole-filling for finding boundaries
- Binarizing the image
- Defining a seed and an ellipse kernel for closing operation
- Defining a function to fill the remaining holes
- Finding the boundaries by subtracting the recent image and its eroded version

## HW6: Registration and Segmentation
### 1. Hough transform for segmentation
- Tuning the thresholds for the canny filter
- Using the mentioned thresholds in the hough transform algorithm

### 2. Feature-based registration
- Asking the user for entering 3 corresponding points in each image
- Giving the points as input to cv.getAffineTransform() which returns the transformation matrix
