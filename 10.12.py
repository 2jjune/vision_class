import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from random import randint

# image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/Lena_color.png').astype(np.float32)/255.
#
# #-------------------k-means clustering------------------------------
# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#
# data = image_lab.reshape((-1,3))
# num_classes = 8
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
# _, labels, centers = cv2.kmeans(data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
# segmented_lab = centers[labels.flatten()].reshape(image.shape)
# segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2RGB)
#
# plt.subplot(121)
# plt.axis('off')
# plt.title('original')
# plt.imshow(image[:,:,[2,1,0]])
# plt.subplot(122)
# plt.axis('off')
# plt.title('segmented')
# plt.imshow(segmented)
# plt.show()

#---------------------
# image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/Lena_color.png')
# show_img = np.copy(image)
# seeds = np.full(image.shape[0:2], 0, np.int32)
# segmentation = np.full(image.shape, 0, np.uint8)
#
# n_seeds = 9
#
# colors = []
# for m in range(n_seeds):
#     colors.append((255*m/n_seeds, randint(0,255),randint(0,255)))
#
# mouse_pressed = False
# current_seed=1
# seeds_updated = False
#
# def mouse_callback(event, x, y, flags, param):
#     global mouse_pressed, seeds_updated
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         mouse_pressed=True
#         cv2.circle(seeds, (x,y), 5, (current_seed), cv2.FILLED)
#         cv2.circle(show_img, (x,y), 5, colors[current_seed-1], cv2.FILLED)
#         seeds_updated=True
#
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if mouse_pressed:
#             cv2.circle(seeds, (x, y), 5, (current_seed), cv2.FILLED)
#             cv2.circle(show_img, (x,y), 5, colors[current_seed-1], cv2.FILLED)
#             seeds_updated=True
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         mouse_pressed = False
#
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', mouse_callback)
#
# while True:
#     cv2.imshow('segmentation', segmentation)
#     cv2.imshow('image', show_img)
#
#     k=cv2.waitKey(1)
#
#     if k==27:
#         break
#     elif k == ord('c'):
#         show_img=np.copy(image)
#         seeds = np.full(image.shape[0:2], 0, np.int32)
#         segmentation = np.full(image.shape, 0, np.uint8)
#     elif k>0 and chr(k).isdigit():
#         n=int(chr(k))
#         if 1<=n <= n_seeds and not mouse_pressed:
#             current_seed=n
#
#     if seeds_updated and not mouse_pressed:
#         seeds_copy = np.copy(seeds)
#         cv2.watershed(image, seeds_copy)
#         segmentation = np.full(image.shape, 0, np.uint8)
#         for m in range(n_seeds):
#             segmentation[seeds_copy==(m+1)] = colors[m]
#
#         seeds_updated = False
#
# cv2.destroyAllWindows()

#----------------grabcut--------------------------
img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/Lena_color.png')
show_img = np.copy(img)

mouse_pressed = False
y = x = w = h = 0

def mouse_callback(event, _x, _y, flags, param):
    global show_img,x,y,w,h,mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed=True
        x,y=_x,_y
        show_img=np.copy(img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            show_img = np.copy(img)

            cv2.rectangle(show_img, (x,y), (_x,_y), (0,255,0), 3)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        w,h=_x-x,_y-y
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', show_img)
    k=cv2.waitKey(1)

    if k == ord('a') and not mouse_pressed:
        if w*h>0:
            break
cv2.destroyAllWindows()

labels = np.zeros(img.shape[:2], np.uint8)
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, (x,y,w,h), None, None, 5, cv2.GC_INIT_WITH_RECT)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD) | (labels==cv2.GC_BGD)] //= 3

cv2.imshow('image', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

label = cv2.GC_BGD
lbl_clrs = {cv2.GC_BGD:(0,0,0), cv2.GC_FGD:(255,255,255)}

def mouse_callback(event, x, y, flags, param):
    global mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed=True
        cv2.circle(labels, (x,y), 5, label, cv2.FILLED)
        cv2.circle(show_img, (x,y), 5, lbl_clrs[label], cv2.FILLED)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
            cv2.circle(show_img, (x,y), 5, lbl_clrs[label], cv2.FILLED)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', show_img)
    k=cv2.waitKey(1)


    if k == ord('a') and not mouse_pressed:
        break
    elif k == ord('l'):
        label = cv2.GC_FGD-label

cv2.destroyAllWindows()

labels, bgdModel, fgdModel = cv2.grabCut(img, labels, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD)|(labels == cv2.GC_BGD)] //=3

cv2.imshow('image',show_img)
cv2.waitKey()
cv2.destroyAllWindows()