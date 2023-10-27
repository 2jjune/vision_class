import cv2
import numpy as np
import imutils

# video = cv2.VideoCapture('./traffic.mp4')
# prev_pts = None
# prev_gray_frame = None
# tracks = None
#
# while True:
#     retval, frame = video.read()
#     if not retval: break
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     if prev_pts is not None:
#         pts, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, prev_pts, None, winSize=(15,15), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         good_pts = pts[status==1]
#         if tracks is None: tracks = good_pts
#         else: tracks = np.vstack((tracks,good_pts))
#         for p in tracks:
#             cv2.circle(frame, (p[0], p[1]), 3, (0,255, 0), -1)
#     else:
#         pts = cv2.goodFeaturesToTrack(gray_frame, 500, 0.05, 10)
#         pts = pts.reshape(-1,1,2)
#     prev_pts = pts
#     prev_gray_frame = gray_frame
#
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1) & 0xff
#     if key ==27: break
#     if key == ord('c'):
#         tracks = None
#         prev_pts = None
# cv2.destroyAllWindows()

def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)
        # if 2<= cv2.norm(delta) <= 10:
            # cv2.arrowedLine(img, pt1[::-1], pt2[::-1], (0,0,255),5,cv2.LINE_AA,0,0.4)
    norm_opt_flow = np.linalg.norm(flow,axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)


    norm_opt_flow = (norm_opt_flow*255).astype(np.uint8)

    img = cv2.medianBlur(img, 13)
    img = cv2.GaussianBlur(img, (13,13), 0)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    img = cv2.morphologyEx(closed, cv2.MORPH_DILATE, (3, 3), iterations=3)
    img = np.clip(img + (img-128)*2,0,255) #명암비 조절
    cv2.imshow('optical flow', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))



    #쓰레쉬홀드
    norm_opt_flow = cv2.medianBlur(norm_opt_flow, 15)
    norm_opt_flow = cv2.GaussianBlur(norm_opt_flow, (21,21), 0)
    # _,norm_opt_flow = cv2.threshold(norm_opt_flow, 253, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    opened = cv2.morphologyEx(norm_opt_flow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    closed = cv2.morphologyEx(norm_opt_flow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    norm_opt_flow = cv2.morphologyEx(closed, cv2.MORPH_DILATE, (5, 5), iterations=3)

    # for i in range(norm_opt_flow.shape[1]):
    #     for j in range(norm_opt_flow.shape[0]):
    #         if norm_opt_flow[j][i]>100:
    #             norm_opt_flow[j][i]=255
    #         else:
    #             norm_opt_flow[j][i]=0

    cv2.imshow('optical flow magnitude', norm_opt_flow)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_opt_flow = cv2.add(norm_opt_flow, np.clip(gray_img-128,0,255))
    # norm_opt_flow = np.where((norm_opt_flow+gray_img)>255,255,norm_opt_flow-255+cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # norm_opt_flow = norm_opt_flow-255+cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, norm_opt_flow = cv2.threshold(norm_opt_flow, 100, 150, cv2.THRESH_BINARY)
    norm_opt_flow = cv2.bilateralFilter(norm_opt_flow, 3, 75, 75)
    cv2.imshow('plus', norm_opt_flow)

    # norm_opt_flow = cv2.medianBlur(norm_opt_flow, 5)
    # norm_opt_flow = cv2.GaussianBlur(norm_opt_flow, (7,7), 0)
    #히스토그램 평탄화

    edged = cv2.Canny(norm_opt_flow, 30, 120)
    cv2.imshow('aaa', edged)
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
        # 원본이미지
        output = norm_opt_flow.copy()
        cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

        # 검은배경
        tmp = np.zeros([norm_opt_flow.shape[0], norm_opt_flow.shape[1], 3], dtype=np.uint8)
        cv2.drawContours(tmp, [findCnt], -1, (255, 255, 255), 2)

        cv2.imwrite('./result.jpg', tmp)
        cv2.imshow('edged', tmp)
        cv2.waitKey()

    k = cv2.waitKey(1)

    if k == 27:
        return 1
    else:
        return 0

cap = cv2.VideoCapture('./True_new_1.mp4')
_, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)
init_flow = True

while True:
    status_cap, frame = cap.read()
    frame = cv2.resize(frame, (0,0), None, 0.5,0.5)
    if not status_cap:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if init_flow:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
    else:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, opt_flow, 0.5, 5, 13, 10, 5, 1.1, cv2.OPTFLOW_USE_INITIAL_FLOW)
    prev_frame = np.copy(gray)
    if display_flow(frame,opt_flow):
        break;

cv2.destroyAllWindows()

# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# _, prev_frame = cap.read()
#
# prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)
#
# flow_DualTVL1 = cv2.createOptFlow_DualTVL1()
#
# while 1:
#     status_cap, frame = cap.read()
#     frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
#     if not status_cap:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if not flow_DualTVL1.getUseInitialFlow():
#         opt_flow = flow_DualTVL1.calc(prev_frame, gray, None)
#         flow_DualTVL1.setUseInitialFlow(True)
#     else:
#         opt_flow = flow_DualTVL1.calc(prev_frame, gray, opt_flow)
#
#     prev_frame = np.copy(gray)
#
#     if display_flow(frame, opt_flow):
#         break;
#
# cv2.destroyAllWindows()








###파노라마

# images = []
# images.append(cv2.imread('./data/panorama/0.jpg', cv2.IMREAD_COLOR))
# images.append(cv2.imread('./data/panorama/1.jpg', cv2.IMREAD_COLOR))
# stitcher = cv2.createStitcher()
# ret, pano = stitcher.stitch(images)
#
# if ret == cv2.STITCHER_OK:
#     pano = cv2.resize(pano, dsize=(0,0), fx=0.2, fy=0.2)
#     cv2.imshow('pano',pano)
#     cv2. waitKey()
#     cv2.destroyAllWindows()
# else:
#     print('error')