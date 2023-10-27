import argparse
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='C:/Users/dlwld/PycharmProjects/vision/data/Lena_color.png')
parser.add_argument('--out_png', default='C:/Users/dlwld/PycharmProjects/vision/data/Lena_compressed.png')
parser.add_argument('--out_jpg', default='C:/Users/dlwld/PycharmProjects/vision/data/Lena_compressed.jpg')
params = parser.parse_args()


image = cv2.imread(params.path)
#-------------------------------------- Reading image from file -----------------------------------------------------------
def read_img(img):
    assert img is not None
    print('read {}'.format(params.path))
    print('shape:', img.shape)
    print('dtype:', img.dtype)

    cv2.imshow('aa',img)
    cv2.waitKey()
    img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)

    assert img is not None
    print('read {} as grayscale'.format(params.path))
    print('shape:', img.shape)
    print('dtype:', img.dtype)


#--------------------------------------------------------------------------------------------------------------------------


#-------------------------------------- Scrollbars in OpenCV window---------------------------------------------------------
def scrollbar():
    cv2.namedWindow('window')
    fill_val = np.array([255,255,255], np.uint8)

    def trackbar_callback(idx, value):
        fill_val[idx] = value

    cv2.createTrackbar('R','window', 255, 255, lambda v: trackbar_callback(2,v))
    cv2.createTrackbar('G','window', 255, 255, lambda v: trackbar_callback(1,v))
    cv2.createTrackbar('B','window', 255, 255, lambda v: trackbar_callback(0,v))

    while True:
        image = np.full((500,500,3), fill_val)
        cv2.imshow('window', image)
        key = cv2.waitKey(3)
        if key == 27:
            break

    cv2.destroyALLWindows()
#--------------------------------------------------------------------------------------------------------------------------



#--------------------------- Drawing 2D primitives -----------Handling user input from mouse--------------------------------
def rand_pt(w, h, mult=1.):
    return (random.randrange(int(w*mult)), random.randrange(int(h*mult)))

def drawing_2d_handling(image):
    w, h = image.shape[1], image.shape[0]

    cv2.circle(image, rand_pt(w,h), 40, (255,0,0))
    cv2.circle(image, rand_pt(w,h), 5, (255,0,0), cv2.FILLED)
    cv2.circle(image, rand_pt(w,h), 40, (255,85,85),2)
    cv2.circle(image, rand_pt(w,h), 40, (255,170,170), 2, cv2.LINE_AA)
    cv2.line(image, rand_pt(w,h), rand_pt(w,h), (0,255,0))
    cv2.line(image, rand_pt(w,h), rand_pt(w,h), (85,255,85), 3)
    cv2.line(image, rand_pt(w,h), rand_pt(w,h), (170,255,170), 3, cv2.LINE_AA)
    cv2.arrowedLine(image, rand_pt(w,h), rand_pt(w,h), (0,0,255), 3, cv2.LINE_AA)
    cv2.rectangle(image, rand_pt(w,h), rand_pt(w,h), (255,255,0),3)
    cv2.ellipse(image, rand_pt(w,h), rand_pt(w,h,0.3), random.randrange(360),0,360, (255,255,255), 3)
    cv2.putText(image, 'OpenCV', rand_pt(w,h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    cv2.imshow('result', image)
    key = cv2.waitKey(0)

mouse_pressed = False
s_x = s_y = e_x = e_y = -1
# image_to_show = np.copy(image)
w, h = image.shape[1], image.shape[0]

def mouse_callback(event, x, y, flags, param):
    global image, s_x, s_y, e_x, e_y, mouse_pressed, image_to_show, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x,y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            tmp = random.randint(0,2)
            # cv2.rectangle(image_to_show, (s_x,s_y), (x,y),(0,255,0),1)
            # cv2.line(image_to_show, (s_x,s_y), (x,y), (0, 255, 0))
            cv2.arrowedLine(image_to_show, (s_x,s_y), (x,y), (0, 0, 255), 3, cv2.LINE_AA)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x,y
    cv2.imshow('image',image_to_show)

def mouse(image):
    global image_to_show
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)
    while True:
        cv2.imshow('image', image_to_show)
        k = cv2.waitKey(1)
        if k == ord('c'):
            if s_y>e_y:
                s_y, e_y = e_y, s_y
            if s_x>e_x:
                s_x, e_x = e_x, s_x
            if e_y - s_y>1 and e_x-s_x>0:
                image = image[s_y:e_y, s_x:e_x]
                image_to_show=np.copy(image)
        elif k==27:
            break
    cv2.destroyALLWindows()

#--------------------------------------------------------------------------------------------------------------------------



#-------------------------------------- Handling user input from keyboard ---------------------------------------------------
def handling(image):
    w, h = image.shape[1], image.shape[0]
    finish = False
    image_to_show = np.copy(image)
    while not finish:
        cv2.imshow('result', image_to_show)
        key = cv2.waitKey(0)
        if key == ord('p'):
            for pt in [rand_pt(w,h) for _ in range(10)]:
                cv2.circle(image_to_show, pt, 3, (255,0,0), -1)
        elif key == ord('l'):
            cv2.line(image_to_show, rand_pt(w, h), rand_pt(w, h), (0, 255, 0),3)
        elif key == ord('r'):
            cv2.rectangle(image_to_show, rand_pt(w, h), rand_pt(w, h), (0, 0, 255), 3)
        elif key == ord('e'):
            cv2.ellipse(image_to_show, rand_pt(w, h), rand_pt(w, h), random.randrange(360), 0, 360, (255, 255, 0), 3)
        elif key == ord('t'):
            cv2.putText(image_to_show, 'OpenCV', rand_pt(w, h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        elif key == ord('c'):
            image_to_show = np.copy(image)
        elif key == ord('w'):
            cv2.imwrite('C:/Users/dlwld/PycharmProjects/vision/data/Lena_draw.png', image_to_show)
        elif key == ord('a'):
            cv2.arrowedLine(image_to_show, rand_pt(w, h), rand_pt(w, h), (0, 200, 255), 3, cv2.LINE_AA)

        elif key == 27:
            finish=True
#--------------------------------------------------------------------------------------------------------------------------



#---------------------------------- Saving image using lossy and lossless compression----------------------------------------
def saving(img):
    cv2.imwrite(params.out_png, img, [cv2.IMWRITE_PNG_COMPRESSION,0])
    saved_img = cv2.imread(params.out_png)
    assert saved_img.all() == img.all()

    cv2.imwrite(params.out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY,0])

#--------------------------------------------------------------------------------------------------------------------------

#1번
# read_img(image)

#2번
# mouse(image)

#3번, 4번, 5번
# handling(image)


# saving(img)



# image_to_show = np.copy(image)

image = cv2.imread(params.path)
image_to_show = np.copy(image)
w, h = image.shape[1], image.shape[0]
while(1):

    cv2.imshow('image', image_to_show)
    # cv2.imshow('aa', image)
    # cv2.waitKey(0)
    # cv2.destroyALLWindows()


    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    # while True:
    #     cv2.imshow('image', image_to_show)
    #     k = cv2.waitKey(1)
    #     if k == ord('c'):
    #         if s_y > e_y:
    #             s_y, e_y = e_y, s_y
    #         if s_x > e_x:
    #             s_x, e_x = e_x, s_x
    #         if e_y - s_y > 1 and e_x - s_x > 0:
    #             image = image[s_y:e_y, s_x:e_x]
    #             image_to_show = np.copy(image)
    #     if k == 27:
    #         # cv2.destroyALLWindows()
    #         break
    key = cv2.waitKey(0)


    # cv2.imshow('result', image_to_show)
    if key == ord('p'):
        for pt in [rand_pt(w, h) for _ in range(10)]:
            cv2.circle(image_to_show, pt, 3, (255, 0, 0), -1)
    elif key == ord('l'):
        cv2.line(image_to_show, rand_pt(w, h), rand_pt(w, h), (0, 255, 0), 3)
    elif key == ord('r'):
        cv2.rectangle(image_to_show, rand_pt(w, h), rand_pt(w, h), (0, 0, 255), 3)
    elif key == ord('e'):
        cv2.ellipse(image_to_show, rand_pt(w, h), rand_pt(w, h), random.randrange(360), 0, 360, (255, 255, 0), 3)
    elif key == ord('t'):
        cv2.putText(image_to_show, 'OpenCV', rand_pt(w, h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    elif key == ord('c'):
        image_to_show = np.copy(image)
    elif key == ord('w'):
        cv2.imwrite('C:/Users/dlwld/PycharmProjects/vision/data/Lena_draw.png', image_to_show)
    elif key == ord('a'):
        cv2.arrowedLine(image_to_show, rand_pt(w, h), rand_pt(w, h), (0, 200, 255), 3, cv2.LINE_AA)

    if key == 27:
        break