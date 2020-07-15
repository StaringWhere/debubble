import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
from time import time

def pitFilter(img, do_show):
    # 滤除图像中的小白点

    # ---------------中值滤波------------------
    # start = time()
    # img_medianBlur=cv2.medianBlur(img, 11)
    # print('Time Spend: Median blur ', time() - start)

    # ---------------补洞算法------------------
    start = time()

    # 转换为亮度和饱和度图像
    img_L = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1]
    # img_L = img[:,:,2]
    img_S = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]

    # 根据饱和度生成mask
    ret, mask1 = cv2.threshold(img_S, 80, 255, cv2.THRESH_BINARY_INV)

    # 根据相对亮度生成mask
    mask2 = cv2.adaptiveThreshold(img_L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -4)
    # 滤去大面积目标，即物体边缘
    area_th = 10 # 面积阈值
    retval, labels, stats, centroids=cv2.connectedComponentsWithStats(mask2, connectivity=8, ltype=cv2.CV_32S)
    areas = stats[:,4]
    # print(sum(areas > area_th))
    for index, area in enumerate(areas):
        if(area > area_th):
            labels[labels == index] = 0
    mask2 = ((labels != 0) * 255).astype('uint8')

    # 两mask取交集，进一步缩小误判
    mask = cv2.bitwise_and(mask1, mask2, dst=None, mask=None)

    # mask膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask2, kernel, iterations=1)

    # 补洞
    img_inpaint = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)

    print('Time Spend: Inpaint ', time() - start)

    # ----------------------保存-----------------------
    do_save = 0
    if do_save:
        # cv2.imwrite('2_img_medianBlur.jpg', img_medianBlur)
        cv2.imwrite('2_inpaint.jpg', img_inpaint)

    # ----------------------展示-----------------------
    if do_show:
        # cv2.imshow('Median Blur',img_medianBlur)
        # cv2.imshow('mask1',mask1)
        cv2.imshow('mask2',mask2)
        cv2.imshow('mask',mask)
        cv2.imshow('original',img)
        cv2.imshow('Inpaint',img_inpaint)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_inpaint, mask