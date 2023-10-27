import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/Lena.png',0)

cv2.imshow('org', image)
cv2.waitKey()

# unsharp mask
KSIZE = 11
ALPHA = 2
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1+ALPHA
filtered = cv2.filter2D(image, -1, kernel)

cv2.imshow('unsharp', filtered)
cv2.waitKey()

# sobel
dx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1)


cv2.imshow('dx', dx)
cv2.imshow('dy', dy)
# cv2.imshow('xy', xy)
cv2.waitKey()

# gabor filter
kernel = cv2.getGaborKernel((21,21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel*kernel).sum())
filtered_gabor = cv2.filter2D(filtered,-1,kernel)
cv2.imshow('gabor', filtered_gabor)
cv2.waitKey()

dff = dx+dy-filtered_gabor
# threshold 변화
def onChange(pos):
    pass
cv2.namedWindow("Trackbar Windows")
cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)
cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)
while cv2.waitKey(1) != ord('q'):
    thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
    maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")
    _, binary = cv2.threshold(dff, thresh, maxval, cv2.THRESH_BINARY)
    cv2.imshow("Trackbar Windows", binary)
cv2.destroyAllWindows()

# thr, mask = cv2.threshold(binary, 200,1,cv2.THRESH_BINARY)
# adapt_mask = cv2.adaptiveThreshold(binary,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
#
# cv2.imshow('threshold', adapt_mask)
# cv2.waitKey()

# opening, closing
# _, binary = cv2.threshold(image, -1,1,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3,3), iterations=10)
# dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3,3), iterations=10)

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
grad = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

cv2.imshow('opening', opened)
cv2.imshow('closing', closed)
cv2.waitKey()
cv2.destroyAllWindows()

# frequency-based filtering
image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/Lena.png',0).astype(np.float32)/255
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0,1])
sz=25
mask = np.zeros(fft.shape, np.uint8)
mask[image.shape[0]//2-sz:image.shape[0]//2+sz,
    image.shape[1]//2-sz:image.shape[1]//2+sz,:] =1
fft_shift*=mask
fft = np.fft.ifftshift(fft_shift,axes=[0,1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))

cv2.imshow('dff', filtered)
# cv2.imshow('dff2', mask_new)
cv2.waitKey()

# 원,사각형 필터링
circle_image = np.zeros((512,512), np.uint8)
cv2.circle(circle_image, (250,250),100,255,-1)
rect_image = np.zeros((512,512), np.uint8)
cv2.rectangle(rect_image,(100,100),(400,250),255,-1)

circle_and_rect_image = circle_image & rect_image
circle_or_rect_image = circle_image | rect_image

fft2 = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift2 = np.fft.fftshift(fft2, axes=[0,1])
mask2 = circle_or_rect_image
# mask2 = np.expand_dims(mask2, axes=[0,1])
# mask2[circle_or_rect_image.shape[0]//2-sz:circle_or_rect_image.shape[0]//2+sz,
#     circle_or_rect_image.shape[1]//2-sz:circle_or_rect_image.shape[1]//2+sz,:] = 1
print(fft_shift2.shape)
print(fft_shift2[:,:,0])
print(fft_shift2[:,:,1])
print(mask2.shape)
# fft_shift2[:,:,0]*=mask2
fft_shift2[:,:,1]*=mask2
fft2 = np.fft.ifftshift(fft_shift2,axes=[0,1])

filtered2 = cv2.idft(fft2, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# cv2.imshow('circle_rect', circle_and_rect_image)
cv2.imshow('circle_rect2', filtered2)
cv2.waitKey()