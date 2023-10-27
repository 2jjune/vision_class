import cv2, numpy as np
import argparse
import matplotlib.pyplot as plt

# image = np.full((480,640,3),255,np.uint8)
# cv2.imshow('white', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# image = np.full((480,640,3),(0,0,255),np.uint8)
# cv2.imshow('red', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
#
# image.fill(0)
# cv2.imshow('black', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# image[240,160] = image[240,320] = image[240,480] = (255,255,255)
# cv2.imshow('black with white pixels', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# image[:,:,0] = 255
# cv2.imshow('blue with white pixels', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# image[:,320,:] = 255
# cv2.imshow('blue with white line', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# image[100:600,100:200,2] = 255
# cv2.imshow('image', image)
# cv2.waitKey()
# cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='C:/Users/dlwld/PycharmProjects/vision/data/test5-1.jpg')
params = parser.parse_args()

def run_histogram_equalization(image_path):
    rgb_img = cv2.imread(image_path)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    cv2.imshow('equalized_img', equalized_img)
    cv2.waitKey(0)

run_histogram_equalization(params.path)


image = cv2.imread(params.path)
cv2.imshow('color_lena', image)
cv2.waitKey()
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_lena', gray)
cv2.waitKey()
cv2.destroyAllWindows()
#-----------------------
hist, bins = np.histogram(gray,256,[0,255])
gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0,255])

cv2.imshow('equalized grey', gray_eq)
cv2.waitKey()
#--------------------------
gamma = 0.5
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255

corrected_image = np.power(gray, gamma)
cv2.imshow('corrected_image', corrected_image)
cv2.waitKey()

cv2.destroyAllWindows()

#------------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsv = hsv/255.
h, s, v = cv2.split(hsv)

cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)
cv2.waitKey()
# cv2.destroyAllWindows()

median = cv2.medianBlur(h, 7)
cv2.imshow('median',median)
cv2.waitKey()

gauss = cv2.GaussianBlur(s, (7,7), 0)
cv2.imshow('gauss',gauss)
cv2.waitKey()

bilat = cv2.bilateralFilter(v, -1, 0.3, 10)
cv2.imshow('bilat',bilat)
cv2.waitKey()