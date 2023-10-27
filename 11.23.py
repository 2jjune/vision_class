import cv2
import numpy as np
import os


# pattern_size = (10,7)
# samples= []
# file_list = os.listdir('./data/pinhole_calib/')
#
# img_file_list = [file for file in file_list if file.startswith('img')]
#
# for filename in img_file_list:
#     frame = cv2.imread(os.path.join('./data/pinhole_calib/', filename))
#     res, corners = cv2.findChessboardCorners(frame, pattern_size)
#
#     img_show = np.copy(frame)
#     cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
#     cv2.putText(img_show, 'samples captured: %d'%len(samples), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0)), 2
#     cv2.imshow('chess', img_show)
#
#     wait_time = 0 if res else 30
#     k = cv2.waitKey(wait_time)
#
#     if k == ord('s') and res:
#         samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
#     elif k==27:
#         break
# cv2.destroyAllWindows()
#
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
#
# for i in range(len(samples)):
#     img, corners = samples[i]
#     corners = cv2.cornerSubPix(img, corners, (10,10), (-1,-1), criteria)
#
# pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
# pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1,2)
#
# images, corners = zip(*samples)
# pattern_points = [pattern_points]*len(corners)
#
# rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(pattern_points, corners, images[0].shape, None, None)
#
# np.save('camera_mat.npy', camera_matrix)
# np.save('dist_coefs.npy', dist_coefs)
#
# print(np.load('camera_mat.npy'))
# print(np.load('dist_coefs.npy'))




# camera_matrix = np.load('./data/pinhole_calib/camera_mat.npy')
# dist_coefs = np.load('./data/pinhole_calib/dist_coefs.npy')
#
# img = cv2.imread('./data/pinhole_calib/img_00.png')
# pattern_size = (10,7)
# res, corners = cv2.findChessboardCorners(img, pattern_size)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
# corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (10,10), (-1,-1), criteria)
#
# h_corners = cv2.undistortPoints(corners, camera_matrix, dist_coefs)
# h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]
#
# img_pts, _ = cv2.projectPoints(h_corners, (0,0,0), (0,0,0), camera_matrix, None)
#
# for c in corners:
#     cv2.circle(img, tuple(c[0]), 10, (0,255,0), 2)
#
# for c in img_pts.squeeze().astype(np.float32):
#     cv2.circle(img, tuple(c), 5, (0,0,255), 2)
#
# cv2.imshow('undistorted', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# img_pts, _ = cv2.projectPoints(h_corners, (0,0,0), (0,0,0), camera_matrix, dist_coefs)
#
# for c in img_pts.squeeze().astype(np.float32):
#     cv2.circle(img, tuple(c), 2, (255,255,0), 2)
#
# cv2.imshow('reprojected', img)
# cv2.waitKey()
# cv2.destroyAllWindows()





camera_matrix = np.load('./data/pinhole_calib/camera_mat.npy')
dist_coefs = np.load('./data/pinhole_calib/dist_coefs.npy')
img = cv2.imread('./data/pinhole_calib/img_00.png')


cv2.imshow('org', img)

ud_img = cv2.undistort(img, camera_matrix, dist_coefs)
cv2.imshow('undistorted 1', ud_img)

opt_cam_mat, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, img.shape[:2][::-1], 0)
ud_img = cv2.undistort(img, camera_matrix, dist_coefs, None, opt_cam_mat)
cv2.imshow('undistorted 2', ud_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
