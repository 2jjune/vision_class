import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

#------------------1번-----------------
image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/TestImage/Lena.png')

rgb = input("r,g,b 중 하나를 입력하세요 : ")

if rgb == 'r':
    hist, bins = np.histogram(image[:, :, 2], 256, [0, 255])
    plt.fill(hist)
    plt.xlabel('pixel value')
    plt.show()
    r = cv2.equalizeHist(image[:, :, 2])
    hist, bins = np.histogram(r, 256, [0, 255])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value of r')
    plt.show()
    image[:, :, 2] = r

elif rgb == 'g':
    hist, bins = np.histogram(image[:, :, 1], 256, [0, 255])
    plt.fill(hist)
    plt.xlabel('pixel value')
    plt.show()
    g = cv2.equalizeHist(image[:, :, 1])
    hist, bins = np.histogram(g, 256, [0, 255])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value of g')
    plt.show()
    image[:, :, 1] = g

elif rgb == 'b':
    print(image[:,:,0])
    hist, bins = np.histogram(image[:, :, 0], 256, [0, 255])
    plt.fill(hist)
    plt.xlabel('pixel value')
    plt.show()
    b = cv2.equalizeHist(image[:, :, 0])
    hist, bins = np.histogram(b, 256, [0, 255])
    plt.fill_between(range(256), hist, 0)
    plt.xlabel('pixel value of b')
    plt.show()
    image[:, :, 0] = b


cv2.imshow('aa',image)
cv2.waitKey()
cv2.destroyAllWindows()

#----------------2번----------------
image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/TestImage/Lena.png',0)
print(image.shape)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image[i][j] += random.randint(0,50)
        if image[i][j]>=255:
            image[i][j]=255

diameter = input('diameater(-1) : ')
sigmacolor = input('sigmacolor(0.3) : ')
sigmaspace = input('sigmaspace(10) : ')

bilat = cv2.bilateralFilter(image, int(diameter), float(sigmacolor), int(sigmaspace))
cv2.imshow('org',image)
cv2.imshow('bilat',bilat)
cv2.waitKey()
cv2.destroyAllWindows()

#----------------3번----------------
image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/TestImage/Lena.png',0).astype(np.float32)/255
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0,1])
magnitude = cv2.magnitude(fft_shift[:,:,0], fft_shift[:, :, 1]) # 벡터 크기 계산
magnitude = np.log(magnitude)

plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout(True)
plt.show()

radius = int(input('반지름을 입력하세요 : '))
low_high = input('filter를 입력하세요(l, h) : ')

if low_high == 'l':
    mask = np.zeros(fft.shape, np.uint8)
    cv2.circle(mask,(image.shape[1]//2,image.shape[0]//2),radius, (1,1), -1)
elif low_high == 'h':
    mask = np.ones(fft.shape, np.uint8)
    cv2.circle(mask,(image.shape[1]//2,image.shape[0]//2),radius, (0,0), -1)

print(mask[240:250,240:250,0])
print(mask[240:250,240:250,1])
print(mask.shape)

fft_shift*=mask
fft = np.fft.ifftshift(fft_shift,axes=[0,1])


filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))

plt.subplot(121)
plt.axis('off')
plt.imshow(image, cmap = 'gray')

plt.subplot(122)
plt.axis('off')
plt.imshow(filtered, cmap='gray')
plt.tight_layout(True)

# plt.subplot(133)
# plt.axis('off')
# plt.imshow(mask_new*255, cmap = 'gray')
plt.show()


#----------------4번----------------
image = cv2.imread('C:/Users/dlwld/PycharmProjects/vision/TestImage/Lena.png',0)
_, binary = cv2.threshold(image, -1,1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

choice = input('(erosion,dilation,opening,closing) 여부를 선택하세요(e,d,o,c) : ')
num = int(input('반복 횟수를 입력하세요 : '))

if choice == 'e':
    eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, (3,3), iterations=num)
    plt.imshow(eroded, cmap='gray')
    plt.show()
elif choice == 'd':
    dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, (3,3), iterations=num)
    plt.imshow(dilated, cmap='gray')
    plt.show()
elif choice == 'o':
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=num)
    plt.imshow(opened, cmap='gray')
    plt.show()
elif choice == 'c':
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=num)
    plt.imshow(closed, cmap='gray')
    plt.show()




