from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import imutils
import cv2
import numpy as np
import os

# def make_scan_image(image, width, ksize=(5, 5), min_threshold=75, max_threshold=200):
def find_contour(image, width, ksize, min_threshold, max_threshold):
    org_image = image.copy()
    image = imutils.resize(image, width=width)
    ratio = org_image.shape[1] / float(image.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    blurred = cv2.GaussianBlur(gray, ksize, 0)

    # bilat = cv2.bilateralFilter(blurred, 100, 10, 80)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    eroded = cv2.morphologyEx(closed, cv2.MORPH_DILATE, (3, 3), iterations=3)

    edged = cv2.Canny(eroded, min_threshold, max_threshold)
    print('''''''''''''''''')
    print(type(edged))
    print(edged.shape)
    cv2.imshow('aaa', eroded)
    cv2.imshow('aaa2', edged)
    cv2.waitKey()


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
        output = image.copy()
        cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

        #검은배경
        tmp = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
        cv2.drawContours(tmp, [findCnt], -1, (255, 255, 255), 2)

        cv2.imwrite('./result.jpg', output)
        cv2.imshow('edged', output)
        cv2.waitKey()


        # 이미지를 보정
        transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

        return transform_image

    else:
        return 0

def main():
    base_dir = 'C:/Users/dlwld/PycharmProjects/vision/project_data/'
    img_list = os.listdir(base_dir)

    for img in img_list:
        org_image = cv2.imread(base_dir+img)

        receipt_image = find_contour(org_image, 200, (7, 7), 20, 100)

        # cv2.imwrite('C:/Users/USER/PycharmProjects/detect_coordinates_with_yolo/croppedImage/warped/{}_cropped.jpg'.format(img), receipt_image)


main()