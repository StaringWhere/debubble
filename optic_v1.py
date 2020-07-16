import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time

'''
=====================光流法v1========================

---------------光流法v1.1 单stride-------------------

稀疏光流获取背景位移 + 单stride稠密光流找出白点 + hsv过滤 + 前后帧补洞 + 补洞算法补洞 + 物体保护

步骤：

找到需要修复的点：
1. 利用稀疏光流追踪背景的角点（特征是一直慢速），计算出背景的位移
2. 假设当前处理第i帧图像，将第i+stride帧根据背景位移移回原来的位置
3. 将两帧图像用稠密光流算出每个点的位移，距离超过背景位移的视为需要修复的点
4. 用饱和度和亮度配合找到明显且移动缓慢的白点

修复：
1. 利用前后10帧中无需修复的点，根据背景的位移填补当前帧
2. 利用（宽松和严格的）饱和度、明度和大小从原图像中筛选出需要保护的物体（小鱼），从剩余mask中排除掉
3. 剩余待修复的少量区域用补洞算法修复
4. 将小鱼复原

优点：
1. 用前后帧修补图像使得图像连贯性好
2. 对移动的物体效果好，效率高
3. 对背景只存在整体小范围位移的视频效果好
缺点：
1. 有些小白点几乎不动，需要用其他检测算法弥补
2. 背景图像有变形、缩放时修补误差大

待解决的问题：
1. 移速缓慢且不明显的小白点如何去除

---------------光流法v1.2 多stride-------------------

稀疏光流获取背景位移 + 多stride稠密光流找出白点 + 前后帧补洞 + 补洞算法补洞 + 物体保护

步骤：

找到需要修复的点：
1. 利用稀疏光流追踪背景的角点（特征是一直慢速），计算出背景的位移
2. 假设当前处理第i帧图像，将第i+stride帧根据背景位移移回原来的位置
3. 将两帧图像用稠密光流算出每个点的位移，距离超过背景位移的视为需要修复的点mask
4. 将多个stride（如3和30）所生成的mask叠加

修复：
1. 利用前后10帧中无需修复的点，根据背景的位移填补当前帧
2. 利用（宽松和严格的）饱和度、明度和大小从原图像中筛选出需要保护的物体（小鱼），从剩余mask中排除掉
3. 剩余待修复的少量区域用补洞算法修复
4. 复原小鱼

优点：
1. 能捕捉更多缓慢的白点
缺点：
1. 由于第二个步长较大（如30帧），背景缩放不可忽略，导致一些背景一直误判在mask范围内

待解决的问题：
1. 解决背景缩放引起的问题
    解决思路：
    1. 提高阈值(缩放不可忽略，不行)
    2. 修复方法改善为用前后帧平均值修复(好些，指标不治本)
    3. 加入缩放判断(由于角点数量少，在帧跨度小时，偶然误差不可忽略，因此在帧跨度大于10时加入缩放判断)
====================================================

=====================光流法v2.1========================
稀疏光流间接获取帧间位移和缩放 + 多stride稠密光流找出白点 + 前后帧补洞 + 帧平均补洞 + 物体保护

步骤：

找到需要修复的点：
1. 利用稀疏光流追踪背景的角点（特征是一直慢速），计算出相邻帧间的不考虑缩放的位移、考虑缩放的位移、缩放倍数
2. 假设当前处理第i帧图像，stride > 10, 将第i+stride帧根据背景累计位移和累计缩放对齐图像
3. 假设当前处理第i帧图像，stride <= 10, 考虑到缩放的偶然误差，将第i+stride帧根据不考虑缩放的背景累计位移对齐图像
4. 将两帧图像用稠密光流算出每个点的位移，距离超过背景位移的视为需要修复的点mask
5. 将多个stride（如3和30）所生成的mask叠加

修复：
1. 利用前后10帧中无需修复的点，根据背景的位移填补当前帧
2. 利用（宽松和严格的）饱和度、明度和大小从原图像中筛选出需要保护的物体（小鱼），从剩余mask中排除掉
3. 剩余待修复的少量区域用前后帧中去除最高值后的平均值补洞
4. 复原小鱼

优点：
1. 能捕捉更多缓慢的白点
2. 减少了背景的误判
缺点：
1. 尽管去除了最高值，帧平均补洞仍然会残留少量白色拖影
2. 在位移和缩放对齐的过程中，边缘填充的误差会随着帧跨度增加而增加
3. 通过稀疏光流追踪到的角点数量少，难以提取旋转、某一方向上的缩放等变换，能否用其他方式提取特征点
====================================================
'''

def calMovement(src):
    # 读取视频
    cap = cv2.VideoCapture(src)

    # ShiTomasi 角点检测参数
    feature_params = dict(maxCorners=100,
                           qualityLevel=0.3,
                           minDistance=7,
                           blockSize=7)

    # lucas kanade光流法参数
    lk_params = dict(winSize=(15, 15),
                      maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机颜色
    color = np.random.randint(0, 255, (100, 3))

    # 获取第一帧，找到角点
    ret, old_frame = cap.read()
    # 找到原始灰度图
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # 获取图像中的角点，返回到p0中
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 创建一个蒙版用来画轨迹
    mask = np.zeros_like(old_frame)

    indexs = []  # 选取的角点标号
    movements = []  # 移动

    while(1):
        ret, frame = cap.read()
        if type(frame) is type(None):
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

        # 寻找移动小于5的跟踪点
        movement = p1 - p0
        movements.append(movement)
        mag, ang = cv2.cartToPolar(movement[..., 0], movement[..., 1])

        # 选取好的跟踪点
        index = (st == 1) & (mag < 5)
        indexs.append(index.flatten())
        good_new = p1[index]
        good_old = p0[index]

        # 画出轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        # 展示
        # cv2.imshow('frame', img)
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break

        # 更新上一帧的图像和追踪点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # cv2.destroyAllWindows()
    cap.release()

    # 计算位移
    background_movement = []
    for i, movement in enumerate(movements):
        for j in indexs[i:]:
            movement = movement[j]
        movement = np.mean(movement, axis=0)[0]
        background_movement.append(movement)

    return background_movement

def moveImg(img, movement_x, movement_y):
    '''
    平移img，正方向为向右和向下，空缺的地方用原图填充
    '''
    # 取整
    movement_x = round(movement_x)
    movement_y = round(movement_y)
    # 度量尺寸
    width = img.shape[1]
    height = img.shape[0]
    # 平移
    matShift = np.float32([[1, 0, movement_x],[0, 1, movement_y]])
    img_shift = cv2.warpAffine(img, matShift, (width, height))
    img_shift = np.where(img_shift == 0, img, img_shift)

    return img_shift

def calMultiMovement(movement, i, j):
    '''
    从第i帧到第j帧的运动，从第1帧开始计数
    '''
    movement_x = 0
    movement_y = 0
    for k in np.arange(i - 1, j - 1, 1):
        movement_x += movement[k][0]
        movement_y += movement[k][1]

    return movement_x, movement_y

# # 多stride

# def makeMask(movement, first_frame, last_frame, strides, do_plot = 1, mode = 'run'):
#     height, width, channel = cv2.imread('frames/frame_1.jpg').shape

#     # 移动信息图
#     hsv = np.zeros((height, width, channel)).astype('uint8')

#     # 遍历每一行的第1列
#     hsv[..., 1] = 255

#     flow_params = dict(pyr_scale = 0.5,    # 金字塔尺度
#                         levels= 10,        # 金字塔层数
#                         winsize= 2,        # 窗口尺寸，越大对高速运动的物体越容易，但越模糊
#                         iterations= 3,     # 对金字塔每层的迭代次数
#                         poly_n= 5,         # 每个像素中找到多项式展开的邻域像素的大小
#                         poly_sigma= 1.2,   # 高斯标准差，用来平滑倒数
#                         flags= 0           # 光流的方式
#                         )

#     stride_max = max(strides)

#     for i in np.arange(first_frame, last_frame - stride_max + 1, 1):

#         start = time()

#         # --------------------读取图片----------------------
#         frame = cv2.imread('frames/frame_{}.jpg'.format(i))
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         masks = []
#         for stride in strides:
#             frame2 = cv2.imread('frames/frame_{}.jpg'.format(i + stride))
#             if frame2 is None:
#                 break
#             gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#             #--------------------恢复背景位移-------------------
#             # 运动信息
#             movement_x, movement_y = calMultiMovement(movement, i, i + stride)

#             # 根据背景移动信息移动第二张图
#             gray2_shift = moveImg(gray2, -movement_x, -movement_y)

#             #-----------------------稠密光流-------------------------
#             # 返回一个两通道的光流向量，实际上是每个点的像素位移值
#             flow = cv2.calcOpticalFlowFarneback(gray, gray2_shift, flow= None, **flow_params)

#             # 笛卡尔坐标转换为极坐标，获得极轴和极角
#             mag, ang = cv2.cartToPolar(flow[..., 0], flow[...,1])
#             # 光流可视化
#             hsv[..., 0] = ang*180/np.pi/2
#             hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#             rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#             rgb = cv2.add(rgb, frame)

#             #----------------------MASK------------------------------
#             # 用幅度判断
#             mag_bg, ang_bg = cv2.cartToPolar(movement_x, movement_y)
#             mag_bg = mag_bg[0]
#             ang_bg = ang_bg[0]
#             mask = mag - mag_bg > 0 * mag_bg
#             mask = np.where(mag < 2, 0, mask)
#             mask = (mask * 255).astype('uint8')

#             masks.append(mask)

#         mask = np.zeros((height, width)).astype('uint8')
#         for mask2 in masks:
#             mask = cv2.bitwise_or(mask, mask2)

#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#         mask = cv2.dilate(mask, kernel, iterations=1)
#         mask = cv2.erode(mask, kernel, iterations=1)



#         #-----------------------展示-----------------------------
#         if do_plot:
#             cv2.imshow('frame', frame)
#             cv2.imshow('mask', mask)
#             # cv2.imshow('img_inpaint', img_inpaint)

#             if cv2.waitKey(mode == 'run') & 0xff == ord('q'):
#                 break

#         #----------------------保存------------------------------
#         cv2.imwrite('dense/masks_dense/mask_{}.jpg'.format(i), mask)
#         # cv2.imwrite('dense/rgb_dense/rgb_{}.jpg'.format(i), rgb)
#         # cv2.imwrite('dense/inpaint_dense/inpaint_{}.jpg'.format(i), img_inpaint)

#         print(i, time() - start)

#     if do_plot:
#         cv2.destroyAllWindows()


# 单stride + hsv过滤

def makeMask(movement, first_frame, last_frame, stride, do_plot = 1, mode = 'run'):
    height, width, channel = cv2.imread('frames/frame_1.jpg').shape

    # 移动信息图
    hsv = np.zeros((height, width, channel)).astype('uint8')

    # 遍历每一行的第1列
    hsv[..., 1] = 255

    flow_params = dict(pyr_scale = 0.5,    # 金字塔尺度
                        levels= 10,        # 金字塔层数
                        winsize= 2,        # 窗口尺寸，越大对高速运动的物体越容易，但越模糊
                        iterations= 3,     # 对金字塔每层的迭代次数
                        poly_n= 5,         # 每个像素中找到多项式展开的邻域像素的大小
                        poly_sigma= 1.2,   # 高斯标准差，用来平滑倒数
                        flags= 0           # 光流的方式
                        )

    for i in np.arange(first_frame, last_frame - stride + 1, 1):

        start = time()

        # --------------------读取图片----------------------
        frame = cv2.imread('frames/frame_{}.jpg'.format(i))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.imread('frames/frame_{}.jpg'.format(i + stride))
        if frame2 is None:
            break
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        #--------------------恢复背景位移-------------------
        # 运动信息
        movement_x, movement_y = calMultiMovement(movement, i, i + stride)

        # 根据背景移动信息移动第二张图
        gray2_shift = moveImg(gray2, -movement_x, -movement_y)

        #-----------------------稠密光流-------------------------
        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = cv2.calcOpticalFlowFarneback(gray, gray2_shift, flow= None, **flow_params)

        # 笛卡尔坐标转换为极坐标，获得极轴和极角
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[...,1])
        # 光流可视化
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.add(rgb, frame)

        #----------------------MASK------------------------------
        # 用幅度判断
        mag_bg, ang_bg = cv2.cartToPolar(movement_x, movement_y)
        mag_bg = mag_bg[0]
        ang_bg = ang_bg[0]
        mask = mag - mag_bg > 0 * mag_bg
        mask = np.where(mag < 2, 0, mask)
        mask = (mask * 255).astype('uint8')

        # hsv过滤
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ret, mask1 = cv2.threshold(frame_hsv[..., 2], 100, 255, cv2.THRESH_BINARY)
        ret, mask2 = cv2.threshold(frame_hsv[..., 1], 80, 255, cv2.THRESH_BINARY_INV)
        mask_hsv = cv2.bitwise_and(mask1, mask2)

        mask = cv2.bitwise_or(mask_hsv, mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        #-----------------------展示-----------------------------
        if do_plot:
            cv2.imshow('frame', rgb)
            cv2.imshow('mask', mask)
            # cv2.imshow('img_inpaint', img_inpaint)

            if cv2.waitKey(mode == 'run') & 0xff == ord('q'):
                break

        #----------------------保存------------------------------
        cv2.imwrite('dense/masks_dense/mask_{}.jpg'.format(i), mask)
        cv2.imwrite('dense/rgb_dense/rgb_{}.jpg'.format(i), rgb)
        # cv2.imwrite('dense/inpaint_dense/inpaint_{}.jpg'.format(i), img_inpaint)

        print(i, time() - start)

    if do_plot:
        cv2.destroyAllWindows()


def paint(movement, first_frame, last_frame, stride, do_plot = 1, mode = 'run'):
    frames = []
    masks = []
    source = list(np.arange(-6, 7, 1)) # 从其他图片补第0张图片，必须为间距是1的等差数列
    frame_index = source.index(0) # 待修复的帧在数组中的位置
    for i in np.arange(first_frame, last_frame - stride + 1, 1):
        start = time()

        #---------------------读取---------------------
        if i == first_frame:
            for j in source[:-1]:
                frame = cv2.imread('frames/frame_{}.jpg'.format(i + j))
                frames.append(frame)
                mask = cv2.imread('dense/masks_dense/mask_{}.jpg'.format(i + j))
                ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                masks.append(mask)
        
        frame = cv2.imread('frames/frame_{}.jpg'.format(i + source[-1]))
        frames.append(frame)
        mask = cv2.imread('dense/masks_dense/mask_{}.jpg'.format(i + source[-1]))
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask)

        #-----------------------用前后帧补洞----------------------
        inpaint = frames[frame_index].copy()
        mask = masks[frame_index].copy()
        for j in source:
            if j == 0 or masks[frame_index + j] is None:
                continue
            
            if j > 0:
                movement_x, movement_y = calMultiMovement(movement, i, i + j)
            else:
                movement_x, movement_y = calMultiMovement(movement, i + j, i)
                movement_x = -movement_x
                movement_y = -movement_y

            frame2_shift = moveImg(frames[frame_index + j], -movement_x, -movement_y)
            mask2_shift = moveImg(masks[frame_index + j], -movement_x, -movement_y)

            replace_mask = cv2.subtract(mask, mask2_shift)
            inpaint = np.where(replace_mask > 128, frame2_shift, inpaint)
            mask = cv2.subtract(mask, replace_mask)
        

        #----------------------保护大的白色物体--------------------------
        frame = frames[frame_index].copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 宽松条件
        ret, demask1 = cv2.threshold(hsv[..., 2], 200, 255, cv2.THRESH_BINARY)
        ret, demask2 = cv2.threshold(hsv[..., 1], 50, 255, cv2.THRESH_BINARY_INV)
        demask = cv2.bitwise_and(demask1, demask2)

        # 严格条件
        ret, demask1 = cv2.threshold(hsv[..., 2], 230, 255, cv2.THRESH_BINARY)
        ret, demask2 = cv2.threshold(hsv[..., 1], 20, 255, cv2.THRESH_BINARY_INV)
        demask_strict = cv2.bitwise_and(demask1, demask2)

        # 根据严格条件利用面积过滤出物体位置，并获取其在宽松条件mask中的对应连通区域
        obj_label = set()
        _, labels, _, _=cv2.connectedComponentsWithStats(demask, connectivity=8, ltype=cv2.CV_32S)
        area_th = 70 # 面积阈值
        _, labels_strict, stats_strict, _=cv2.connectedComponentsWithStats(demask_strict, connectivity=8, ltype=cv2.CV_32S)
        areas = stats_strict[:,4]
        for index, area in enumerate(areas):
            if(area > area_th and area != areas.max()):
                obj_label = obj_label | set(labels[labels_strict == index])
                break
        
        demask = np.zeros_like(demask)
        for label in obj_label:
            demask = np.where(labels == label, 255, demask)

        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        demask = cv2.dilate(demask, kernel, iterations=1)

        demask = cv2.bitwise_not(demask)
        demask_3d = np.stack((demask, demask, demask), axis = 2)
        
        # #---------------------用周围像素补洞-----------------------
        mask = cv2.bitwise_and(mask, demask_3d)
        inpaint = cv2.inpaint(inpaint, mask[..., 0], 5, cv2.INPAINT_TELEA)

        # 恢复物体
        inpaint = np.where(demask_3d == 0, frame, inpaint)

        #------------------------展示-----------------------------
        if do_plot:
            cv2.imshow('mask_old', masks[frame_index])
            cv2.imshow('old', frames[frame_index])
            cv2.imshow('mask_new', mask)
            cv2.imshow('inpaint', inpaint)
            if cv2.waitKey(mode == 'run') & 0xff == ord('q'):
                break
        
        #------------------------保存-----------------------------
        cv2.imwrite('dense/inpaint_dense/inpaint_{}.jpg'.format(i), inpaint)

        #-----------------------迭代------------------------------
        frames.pop(0)
        masks.pop(0)
        
        print(i, time() - start)

    if do_plot:
        cv2.destroyAllWindows()


# 单stride
stride = 3
first_frame = 15
last_frame = 503

# 获取背景移动信息
movement = calMovement('video.avi')
print('finsh')

makeMask(movement, first_frame, last_frame, stride, mode = 'debug')
# paint(movement, first_frame, last_frame, stride, mode = 'run')


# 多stride
# strides = [3, 20]
# stride_max = max(strides)
# first_frame = 1
# last_frame = 503

# # 获取背景移动信息
# movement = calMovement('video.avi')
# print('finsh')

# makeMask(movement, first_frame, last_frame, strides, mode = 'run')
# paint(movement, first_frame, last_frame, stride_max, mode = 'run')