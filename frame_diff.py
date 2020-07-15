import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
from time import time
from pitFilter import pitFilter

def subimg(img, img_p, img_n, th):
    '''
    利用帧差法计算出快速移动的物体
    -----------  -----------------------
    Parameters   Description
    -----------  -----------------------
    img          当前帧
    img_p        之前的帧
    img_n        之后的帧
    th           帧间灰度差的阈值
    -----------  -----------------------
    '''

    if img.shape != img_p.shape or img.shape != img_n.shape:
        print("subimg error: shape not equal")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    img_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)

    sub_n = img.astype('int') - img_n.astype('int')
    sub_p = img.astype('int') - img_p.astype('int')
    
    sub_n = np.where(sub_n < th, 0, 255)
    sub_p = np.where(sub_p < th, 0, 255)
    sub = cv2.bitwise_and(sub_n, sub_p, dst=None, mask=None)

    return sub.astype('uint8'), sub_p.astype('uint8'), sub_n.astype('uint8')



# 定义一个VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output2.avi',fourcc, 20.0, (1436,664))

for i in np.arange(3,502,1):

    start = time()

    # --------------------读取图片----------------------
    stride = 2 # 帧跨度
    img = cv2.imread('frames/frame_{}.jpg'.format(i))
    img_p = cv2.imread('frames/frame_{}.jpg'.format(i - stride))
    img_n = cv2.imread('frames/frame_{}.jpg'.format(i + stride))

    # ----------------利用帧差法得到mask-----------------
    sub, sub_p, sub_n = subimg(img, img_p, img_n, 15)

    # ----------------用面积滤除白色物体-----------------
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask_obj = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    # 筛选出大面积白色物体
    area_th = 200 # 面积阈值
    retval, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask_obj, connectivity=8, ltype=cv2.CV_32S)
    areas = stats[:,4]
    for index, area in enumerate(areas):
        if(area < area_th):
            labels[labels == index] = 0
    mask_obj = ((labels == 0) * 255).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask_obj = cv2.erode(mask_obj, kernel, iterations=1)
    
    # mask删除大面积白色物体
    mask = cv2.bitwise_and(sub, mask_obj, dst=None, mask=None)

    # ---------------------补洞处理---------------------
    # mask膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # mask = cv2.dilate(mask, kernel, iterations=1)

    # 补洞
    img_inpaint = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    # img_p = cv2.imread('frames/frame_{}.jpg'.format(i - 1))
    # img_n = cv2.imread('frames/frame_{}.jpg'.format(i + 1))
    # mask_3d = np.dstack([mask, mask, mask])
    # img_inpaint = np.where(1, img_p / 2 + img_n / 2, img).astype('uint8')

    
    # 用相对亮度过滤几乎不移动的白点
    filter_more = 0
    if filter_more:
        img_inpaint, _ = pitFilter(img_inpaint, 0)

    #----------------------保存--------------------------
    out.write(img_inpaint)
    # cv2.imwrite('masks/mask_{}.jpg'.format(i), mask)
    cv2.imwrite('mask_{}.jpg'.format(i), mask)

    #----------------------画图--------------------------
    # cv2.imshow('sub', sub)
    # cv2.imshow('sub_p', sub_p)
    # cv2.imshow('sub_n', sub_n)
    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask_obj', mask_obj)
    cv2.imshow('img_inpaint', img_inpaint)
    # cv2.imshow('img_filter', img_filter)
    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

    print(time() - start)

cv2.destroyAllWindows()
# out.release()