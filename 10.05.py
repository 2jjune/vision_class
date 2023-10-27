import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/Lena.png',0)

# cv2.imshow('org', image)
# cv2.waitKey()

#-----------------------threshold----------
otsu_thr, otsu_mask = cv2.threshold(image, 100,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# cv2.imshow('otsu', otsu_mask)
# cv2.waitKey()

plt.figure()
plt.imshow(otsu_mask, cmap='gray')
plt.show()

#-----------------------internal, external----------
# otsu_mask *= 255
# contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
image_external = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)
plt.figure()
plt.subplot(121)
plt.title('external')
plt.imshow(image_external, cmap='gray')

plt.subplot(122)
plt.title('internal')
plt.imshow(image_internal, cmap='gray')

plt.show()

#-----------------------connected component----------
# connectivity = random.randint(1,20)
#
# output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)
# num_labels, labelmap, stats, centers = output
# colored = np.full((otsu_mask.shape[0], otsu_mask.shape[1], 3), 0, np.uint8)
#
# for l in range(1, num_labels):
#     if stats[l][4] > 200:
#         colored[labelmap==l] = (0,255*l/num_labels, 255*(num_labels-1)/num_labels)
#         cv2.circle(colored, (int(centers[l][0]), int(centers[l][1])),5,(random.randint(0,255),random.randint(0,255),random.randint(0,255)),cv2.FILLED)
#
# img = cv2.cvtColor(otsu_mask*255, cv2.COLOR_GRAY2BGR)
# cv2.imshow('connected components', np.hstack((img, colored)))
# cv2.waitKey()
# cv2.destroyAllWindows()
a=0
while (1):
    cv2.imshow('org', image)
    # print(1)

    key = cv2.waitKey(0)

    # print(key)

    if key == ord('p'):
        break
    elif key == ord(' '):
        connectivity = random.randint(1,10)
        print(connectivity)
        output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)
        num_labels, labelmap, stats, centers = output
        colored = np.full((otsu_mask.shape[0], otsu_mask.shape[1], 3), random.randint(0,10), np.uint8)
        print(num_labels, labelmap)
        for l in range(1, num_labels):
            if stats[l][4] > 200:

                colored[labelmap==l] = (0,255*l/num_labels, 255*(num_labels-1)/num_labels)
                cv2.circle(colored, (int(centers[l][0]*random.randint(1,4)), int(centers[l][1]*random.randint(1,4))),15,(random.randint(0,255),random.randint(0,255),random.randint(0,255)),cv2.FILLED)

        img = cv2.cvtColor(otsu_mask*255, cv2.COLOR_GRAY2BGR)
        cv2.imshow('connected components', np.hstack((img, colored)))
        cv2.waitKey()
        a+=1
    if a>4:
        break
cv2.destroyAllWindows()



#-----------------------distance transform----------
# image = np.full((480,640),255,np.uint8)
# cv2.circle(image,(320,240),100,0)
# distmap = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
distmap = cv2.distanceTransform(otsu_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
plt.figure()
plt.imshow(distmap, cmap='gray')
plt.show()



