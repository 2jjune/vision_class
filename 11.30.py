import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import imutils
# p1 = np.eye(3,4, dtype=np.float32)
# p2 = np.eye(3,4, dtype=np.float32)
# p2[0,3]=-1
#
# n=5
# points3d = np.empty((4,n), np.float32)
# points3d[:3,:] = np.random.randn(3,n)
# points3d[3,:] = 1
#
# points1 = p1 @ points3d
# points1 = points1[:2,:]/points1[2,:]
# points1[:2,:] += np.random.randn(2,n)*1e-2
#
# points2 = p2 @ points3d
# points2 = points2[:2,:]/points2[2,:]
# points2[:2,:] += np.random.randn(2,n)*1e-2
#
# points3d_reconstr = cv2.triangulatePoints(p1,p2,points1,points2)
# points3d_reconstr /= points3d_reconstr[3,:]
#
# print('org: ')
# print(points3d[:3].T)
# print('reconstructed: ')
# print(points3d_reconstr[:3].T)

#---
# matplotlib.rcParams.update({'font.size':20})
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#
# data = np.load('C:/Users/dlwld/PycharmProjects/vision/data/stereo/case1/stereo.npy').item()
# Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'],data['Dr'], data['R'], data['T'], data['img_size']
#
# left_img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stereo/case1/left14.png')
# right_img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stereo/case1/right14.png')
#
# R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl,Dl, Kr, Dr, img_size, R, T)
#
# xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl,Dl,R1,Kl, img_size, cv2.CV_32FC1)
# xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr,Dr,R2,Kr, img_size, cv2.CV_32FC1)
#
# left_img_rectified = cv2.remap(left_img, xmap1, ymap1, cv2.INTER_LINEAR)
# right_img_rectified = cv2.remap(right_img, xmap2, ymap2, cv2.INTER_LINEAR)
#
# plt.figure(0, figsize=(12,10))
# plt.subplot(221)
# plt.title('left org')
# plt.imshow(left_img, cmap='gray')
# plt.subplot(222)
# plt.title('right org')
# plt.imshow(right_img, cmap='gray')
# plt.subplot(223)
# plt.title('left rectified')
# plt.imshow(left_img_rectified, cmap='gray')
# plt.subplot(224)
# plt.title('right rectified')
# plt.imshow(right_img_rectified, cmap='gray')
# plt.tight_layout()
# plt.show()

#-------------

# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# data = np.load('C:/Users/dlwld/PycharmProjects/vision/data/stereo/case1/stereo.npy').item()
#
# kl, kr, dl, dr, left_pts, right_pts, e_from_stereo, f_from_stereo = data['Kl'], data['Kr'], data['Dl'],data['Dr'], data['left_pts'], data['right_pts'], data['E'], data['F']
#
# left_pts = np.vstack(left_pts)
# right_pts = np.vstack(right_pts)
#
# left_pts = cv2.undistortPoints(left_pts, kl, dl, P=kl)
# right_pts = cv2.undistortPoints(right_pts, kr, dr, P=kr)
#
# F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)
# E = kr.T @ F @ kl
#
# print('fundamental matrix:')
# print(F)
# print('essential matrix:')
# print(E)


#--------------

matplotlib.rcParams.update({'font.size':20})
# left_img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stereo/left.png')
left_img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/project_test3.jpg')
# right_img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stereo/right.png')
right_img = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/project_test4.jpg')

stereo_bm = cv2.StereoBM_create(32)
dismap_bm = stereo_bm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))
stereo_sgbm = cv2.StereoSGBM_create(0,32)
dismap_sgbm = stereo_sgbm.compute(left_img,right_img)

dismap_sgbm = np.expand_dims(dismap_sgbm,axis=2)
print(dismap_bm.shape)
edged = cv2.Canny(dismap_sgbm, 20, 100)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


findCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        findCnt = approx
        break

if findCnt is not None:
    #원본이미지
    output = dismap_sgbm.copy()
    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

    #검은배경
    tmp = np.zeros([dismap_sgbm.shape[0],dismap_sgbm.shape[1],3],dtype=np.uint8)
    cv2.drawContours(tmp, [findCnt], -1, (255, 255, 255), 2)

    cv2.imwrite('./result.jpg', output)
    cv2.imshow('edged', output)
    cv2.waitKey()


    # 이미지를 보정
    # transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)


plt.figure(0, figsize=(12,10))
plt.subplot(221)
plt.title('left')
plt.imshow(left_img[:,:,[2,1,0]])
plt.subplot(222)
plt.title('right')
plt.imshow(left_img[:,:,[2,1,0]])
plt.subplot(223)
plt.title('BM')
plt.imshow(dismap_bm, cmap='gray')
plt.subplot(224)
plt.title('SGBM')
plt.imshow(dismap_sgbm, cmap='gray')
plt.show()


