import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
from time import time
from pitFilter import pitFilter

def areaFilter(mask, area_th, type):
    '''
    type = 1: 滤去大面积
    type = 2: 滤去小面积
    '''

    retval, labels, stats, centroids=cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
    areas = stats[:,4]
    
    if type == 1:
        for index, area in enumerate(areas):
            if(area > area_th):
                labels[labels == index] = 0
    elif type == 2:
        for index, area in enumerate(areas):
            if(area < area_th):
                labels[labels == index] = 0

    mask_filtered = ((labels != 0) * 255).astype('uint8')

    return mask_filtered


def movement(gray1, gray2, movement_max):
    '''
    计算gray1与gray2的相对位移
    使计算结果movement_x, movement_y尽量满足等式：
        gray2[i + movement_y, j + movement_x] = gray1[i, j]

    @param gray1       : 被对齐的灰度图
    @param gray2       : 对齐的灰度图
    @param movement_max: 最大猜测运动范围
    '''

    if gray1.shape != gray2.shape:
        print('movement error: shape not equal')
        return

    height = gray1.shape[0]
    width = gray1.shape[1]

    # 制作蒙版
    mask = np.ones((height, width)) * 255
    mask[0 : movement_max, :] = 0
    mask[height - movement_max : height, :] = 0
    mask[:, 0 : movement_max] = 0
    mask[:, width - movement_max : width] = 0
    mask = mask.astype('uint8')

    # 最大类间方差图像分割
    _, otsu1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)
    _, otsu2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)

    # 滤去小白点
    otsu1 = areaFilter(otsu1, 100, 2)
    otsu2 = areaFilter(otsu2, 100, 2)

    # 运动信息初始化
    movement_x = 0
    movement_y = 0
    diff_min = 255 * width * height

    # 根据背景帧间差最小值找出运动信息
    for x in np.arange(-movement_max, movement_max + 1, 1):
        for y in np.arange(-movement_max, movement_max + 1, 1):
            # 移位
            matShift = np.float32([[1,0,x],[0,1,y]])
            otsu2_shift = cv2.warpAffine(otsu2, matShift, (width, height))

            # 裁剪边框
            otsu2_shift = cv2.bitwise_and(otsu2_shift, mask)

            # 计算帧间差
            diff = abs(otsu1.astype('int') - otsu2_shift.astype('int'))
            diff_sum = np.sum(diff)

            # 若帧间差变小，更新猜测的移动信息
            if diff_sum < diff_min:
                diff_min = diff_sum
                movement_x = x
                movement_y = y
    
    print(movement_x, movement_y)
    return movement_x, movement_y

def subimg(img1, img2, th):
    '''
    利用帧差法计算出快速移动的物体
    Parameters   Description
    img1         当前帧
    img2         相邻帧
    th           帧间灰度差的阈值
    '''

    if img1.shape != img2.shape:
        print("subimg error: shape not equal")
        return

    height = img1.shape[0]
    width = img1.shape[1]

    # 转灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算背景运动
    movement_x, movement_y = movement(gray1, gray2, 4)
    # 移位
    matShift = np.float32([[1, 0, movement_x],[0, 1, movement_y]])
    gray2_shift = cv2.warpAffine(gray2, matShift, (width, height))
    gray2_shift = np.where(gray2_shift == 0, gray2, gray2_shift)
    
    # 计算帧差
    diff = gray1.astype('int') - gray2_shift.astype('int')
    diff = np.where(diff < th, 0, 255)
    diff = diff.astype('uint8')

    return diff

if __name__ == "__main__":
    i =430
    stride = 2 # 帧跨度
    img = cv2.imread('frames/frame_{}.jpg'.format(i))
    img_p = cv2.imread('frames/frame_{}.jpg'.format(i - stride))
    img_n = cv2.imread('frames/frame_{}.jpg'.format(i + stride))
    diff_p = subimg(img, img_p, 10)
    diff_n = subimg(img, img_n, 10)
    diff = cv2.bitwise_and(diff_p, diff_n)
    cv2.imshow('diff_p', diff_p)
    cv2.imshow('diff_n', diff_n)
    cv2.imshow('diff', diff)
    cv2.imwrite('diff.jpg', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()