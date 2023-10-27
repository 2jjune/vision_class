import cv2
import numpy as np
import imutils

def display_flow(img, flow, stride=40):
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)

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
    norm_opt_flow = np.clip(norm_opt_flow + (norm_opt_flow-200)*2,0,255) #명암비 조절

    norm_opt_flow = cv2.medianBlur(norm_opt_flow, 15)
    norm_opt_flow = cv2.GaussianBlur(norm_opt_flow, (21,21), 0)
    # _,norm_opt_flow = cv2.threshold(norm_opt_flow, 253, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    opened = cv2.morphologyEx(norm_opt_flow, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    closed = cv2.morphologyEx(norm_opt_flow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)
    norm_opt_flow = cv2.morphologyEx(closed, cv2.MORPH_DILATE, (5, 5), iterations=3)

    cv2.imshow('optical flow magnitude', norm_opt_flow)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_opt_flow = cv2.add(norm_opt_flow, np.clip(gray_img-128,0,255))
    norm_opt_flow = cv2.bilateralFilter(norm_opt_flow, 3, 75, 75)
    cv2.imshow('plus', norm_opt_flow)

    edged = cv2.Canny(norm_opt_flow, 20, 100)
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

cap = cv2.VideoCapture('./False_new_1.mp4')
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

