import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

#------------------1번-----------------
image = []
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/boat1.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/budapest1.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/newspaper1.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/s1.jpg'))

# canny edge
for img in image:
    edges = cv2.Canny(img, 200, 100)
    edges = cv2.resize(edges, (800,800))
    cv2.imshow('canny', edges)
    cv2.waitKey()

# haris corner
for img in image:
    print(img.shape)
    corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    show_img = np.copy(img)
    show_img[corners>0.1*corners.max()] = [0,0,255]

    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))

    show_img = cv2.resize(show_img, (800,800))
    cv2.imshow('harris', show_img)
    cv2.waitKey()

cv2.destroyAllWindows()

#------------------2번-----------------
image = []
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/s1.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/s2.jpg'))
# SIFT, SURF, ORB를 추출한 후 매칭 및 RANSAC을 통해서 두 장의 영상간의 homography를 계산하고, 이를 통해 한 장의 영상을 다른 한 장의 영상으로 warping 하는 코드를 작성

orb_detector = cv2.ORB_create(100)
sift_detector = cv2.xfeatures2d.SIFT_create(50)
surf_detector = cv2.xfeatures2d.SURF_create(10000)
surf_detector.setExtended(True)
surf_detector.setNOctaves(3)
surf_detector.setNOctaveLayers(10)
surf_detector.setUpright(False)

#orb
kps0, fea0 = orb_detector.detectAndCompute(image[0], None)
kps1, fea1 = orb_detector.detectAndCompute(image[1], None)

# matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)#SURF
# matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)#SIFT
matches = matcher.match(fea0, fea1)

pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

dbg_img = cv2.drawMatches(image[0], kps0, image[1], kps1, matches, None)

cv2.imshow('ransac_orb', dbg_img[:,:,[0,1,2]])
cv2.waitKey()
dbg_img = cv2.drawMatches(image[0],kps0, image[1], kps1,[m for i,m in enumerate(matches) if mask[i]], None)
cv2.imshow('orb_filtered',dbg_img[:,:,[0,1,2]])
cv2.waitKey()

#sift
kps0, fea0 = sift_detector.detectAndCompute(image[0], None)
kps1, fea1 = sift_detector.detectAndCompute(image[1], None)

# matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)#SURF
# matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)#SIFT
matches = matcher.match(fea0, fea1)

pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

dbg_img = cv2.drawMatches(image[0], kps0, image[1], kps1, matches, None)

cv2.imshow('ransac_sift', dbg_img[:,:,[0,1,2]])
cv2.waitKey()
dbg_img = cv2.drawMatches(image[0],kps0, image[1], kps1,[m for i,m in enumerate(matches) if mask[i]], None)
cv2.imshow('sift_filtered',dbg_img[:,:,[0,1,2]])
cv2.waitKey()

#surf
kps0, fea0 = surf_detector.detectAndCompute(image[0], None)
kps1, fea1 = surf_detector.detectAndCompute(image[1], None)

# matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)#SURF
# matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)#SIFT
matches = matcher.match(fea0, fea1)

pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)

dbg_img = cv2.drawMatches(image[0], kps0, image[1], kps1, matches, None)

cv2.imshow('ransac_surf', dbg_img[:,:,[0,1,2]])
cv2.waitKey()

dbg_img = cv2.drawMatches(image[0],kps0, image[1], kps1,[m for i,m in enumerate(matches) if mask[i]], None)
cv2.imshow('surf_filtered',dbg_img[:,:,[0,1,2]])
cv2.waitKey()

cv2.destroyAllWindows()

l_img = cv2.drawKeypoints(image[0], kps0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
r_img = cv2.drawKeypoints(image[1], kps1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
matcher = cv2.DescriptorMatcher_create('BruteForce')
matches = matcher.knnMatch(fea1, fea0, 2)

hl, wl = image[0].shape[:2]
hr, wr = image[1].shape[:2]
match_points = []
for i in matches:
    if len(i) == 2 and i[0].distance < i[1].distance * 0.75:
        match_points.append((i[0].trainIdx, i[0].queryIdx))

if len(match_points) > 4:
    pts0 = np.float32([kps0[m].pt for (m,_) in match_points])
    pts1 = np.float32([kps1[m].pt for (_,m) in match_points])
    matrix, status = cv2.findHomography(pts1, pts0, cv2.RANSAC, 4.0)
    warped = cv2.warpPerspective(image[1], matrix, (wr + wl, hr))
    warped[0:hl, 0:wl] = image[0]

cv2.imshow('warped',warped)
cv2.waitKey()

cv2.destroyAllWindows()


#------------------3번-----------------
image = []
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/newspaper1.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/newspaper2.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/newspaper3.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/newspaper4.jpg'))

stitcher = cv2.createStitcher()
ret, pano = stitcher.stitch(image)

if ret == cv2.STITCHER_OK:
    pano = cv2.resize(pano, dsize=(0,0), fx=0.2, fy=0.2)
    cv2.imshow('panorama', pano)
    cv2.waitKey()

    cv2.destroyAllWindows()


#------------------4번-----------------
image = []
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/dog_a.jpg'))
image.append(cv2.imread('C:/Users/dlwld/PycharmProjects/vision/data/stitching/dog_b.jpg'))

# -------------4-1
tracks = None

pts = cv2.goodFeaturesToTrack(cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY), 100, 0.05, 10)
pts = pts.reshape(-1,1,2)
prev_pts=pts

prev_gray_frame = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
gray_frame = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
pts, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
good_pts = pts[status==1]
if tracks is None: tracks=good_pts
else: tracks = np.vstack((tracks, good_pts))
for p in tracks:
    cv2.circle(image[1], (p[0],p[1]),3,(0,255,0),-1)
cv2.imshow('lucas-kanade', image[1])
cv2.waitKey()
cv2.destroyAllWindows()
#
#-------------4-2-farneback
def display_flow(frame, opt_flow):
    stride = 40
    for index in np.ndindex(opt_flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i * stride for i in index)
        delta = opt_flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10 * delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(frame, pt1[::-1], pt2[::-1], (0, 0, 255), 5, cv2.LINE_AA, 0, 0.4)

    norm_opt_flow = np.linalg.norm(opt_flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow('optical flow', norm_opt_flow)
    cv2.waitKey()
    cv2.destroyAllWindows()


prev_frame = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)

frame = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)

opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
display_flow(frame,opt_flow)


#-------4-2-tvl1

prev_frame = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)

frame = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)
frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)

flow_DualTVL1 = cv2.createOptFlow_DualTVL1()
if not flow_DualTVL1.getUseInitialFlow():
    opt_flow = flow_DualTVL1.calc(prev_frame,frame,None)
    flow_DualTVL1.setUseInitialFlow(True)
else:
    opt_flow=flow_DualTVL1.calc(prev_frame,frame,opt_flow)

display_flow(frame, opt_flow)


