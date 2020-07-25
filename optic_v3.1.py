import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt

'''
=====================光流法v3.1========================
稀疏光流直接获取帧间单应性矩阵 + 多stride稠密光流找出白点 + hsv过滤 + 前后帧补洞 + 中值滤波补洞 + 物体保护

步骤：

找到需要修复的点：
1. 利用稀疏光流追踪背景的角点（数量远远多于v2）（特征是一直慢速）
2. 假设当前处理第i帧图像，根据第i帧和第i+stride帧的角点计算单应性矩阵，并对齐位置
3. 将两帧图像用稠密光流算出每个点的位移，距离超过背景位移的视为需要修复的点mask
4. 将多个stride（如3和30）所生成的mask叠加
5. 用饱和度和亮度配合找到明显且移动缓慢的白点

修复：
1. 利用前后10帧中无需修复的点，根据背景的位移填补当前帧
2. 利用（宽松和严格的）饱和度、明度和大小从原图像中筛选出需要保护的物体（小鱼），从剩余mask中排除掉
3. 剩余待修复的少量区域用大小为11的中值滤波补洞
4. 复原小鱼

优点：
1. 优化获取帧间变换的方法
2. 改善了帧平均补洞会残留白色拖影的问题，并提高了补洞效率
缺点：
1. 在位移和缩放对齐的过程中，边缘填充的误差会随着帧跨度增加而增加
2. 通过稀疏光流追踪到的角点数量少，难以提取旋转、某一方向上的缩放等变换，能否用其他方式提取特征点
====================================================
'''


def findVertex(src):
    '''
    寻找背景角点
    '''
    # 读取视频
    cap = cv2.VideoCapture(src)

    # ShiTomasi 角点检测参数
    feature_params = dict(maxCorners=1000,
                            qualityLevel=0.01,
                            minDistance=20,
                            blockSize=7)

    # lucas kanade光流法参数
    lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机颜色
    color = np.random.randint(0, 255, (1000, 3))

    for i in range(0):
        _, _ = cap.read()

    # 获取第一帧，找到角点
    ret, old_frame = cap.read()
    # old_frame = old_frame[332: ,718:, :]
    # 找到原始灰度图
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = sharpen(old_gray)
    old_gray = cv2.equalizeHist(old_gray)

    # 获取图像中的角点，返回到p0中
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 创建一个蒙版用来画轨迹
    mask = np.zeros_like(old_frame)

    indexs = []  # 选取的角点标号
    ps = [p0[:,0]] # 保存角点

    while(1):
        ret, frame = cap.read()
        if type(frame) is type(None):
            break
        # frame = frame[332: ,718:, :]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = sharpen(frame_gray)
        frame_gray = cv2.equalizeHist(frame_gray)

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

        # 寻找移动小于5的跟踪点
        movement = p1 - p0
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
        cv2.imshow('frame', img)
        # cv2.imshow('frame_gray', frame_gray)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # 更新上一帧的图像和追踪点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        ps.append(good_new)

    cv2.destroyAllWindows()
    cap.release()

    # 迭代出背景角点 
    background_p = []
    for i, p in enumerate(ps[: -1]):
        for j in indexs[i:]:
            p = p[j]
        background_p.append(p)
    background_p.append(ps[-1])

    return background_p

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    sharpened = cv2.filter2D(img, -1, kernel=kernel)

    return sharpened

def alignImg(vertex, ref, img, ref_id, img_id, fill = 1):
    # 计算单应性矩阵
    h, mask = cv2.findHomography(vertex[img_id - 1], vertex[ref_id - 1], cv2.RANSAC)
    # 变换
    height, width = img.shape[: 2]
    img_reg = cv2.warpPerspective(img, h, (width, height))

    if fill:
        ref_mb = cv2.medianBlur(ref, 21)
        img_reg = np.where(img_reg == 0, ref_mb, img_reg)

    return img_reg

# 多stride

def makeMask(vertex, first_frame, last_frame, strides, do_plot = 1, mode = 'run'):
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

    stride_max = max(strides)

    for i in np.arange(first_frame, last_frame - stride_max + 1, 1):

        start = time()

        # --------------------读取图片----------------------
        frame = cv2.imread('frames/frame_{}.jpg'.format(i))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        masks = []
        for stride in strides:
            frame2 = cv2.imread('frames/frame_{}.jpg'.format(i + stride))
            if frame2 is None:
                break
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            #--------------------恢复背景位移-------------------
            gray2_shift = alignImg(vertex, gray, gray2, i, i + stride)

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
            vec = np.mean(vertex[i + stride - 1] - vertex[i - 1], axis = 0)
            mag_bg = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            mask = mag > mag_bg
            mask = np.where(mag < 2, 0, mask)
            mask = (mask * 255).astype('uint8')

            masks.append(mask)

        mask = np.zeros((height, width)).astype('uint8')
        for mask2 in masks:
            mask = cv2.bitwise_or(mask, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)



        #-----------------------展示-----------------------------
        if do_plot:
            cv2.imshow('frame', frame)
            cv2.moveWindow("frame",0,0)
            cv2.imshow('rgb', rgb)
            cv2.moveWindow("rgb",0,0)
            cv2.imshow('frame2', gray2_shift)
            cv2.moveWindow("frame2",0,0)
            cv2.imshow('mask', mask)
            cv2.moveWindow("mask",0,0)
            # cv2.imshow('img_inpaint', img_inpaint)

            if cv2.waitKey(mode == 'run') & 0xff == ord('q'):
                break

        #----------------------保存------------------------------
        cv2.imwrite('dense/masks_dense/mask_{}.jpg'.format(i), mask)
        # cv2.imwrite('dense/rgb_dense/rgb_{}.jpg'.format(i), rgb)
        # cv2.imwrite('dense/inpaint_dense/inpaint_{}.jpg'.format(i), img_inpaint)

        print(i, time() - start)

    if do_plot:
        cv2.destroyAllWindows()

def paint(vertex, first_frame, last_frame, stride, do_plot = 1, mode = 'run'):
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
        # 修补的图像
        inpaint = frames[frame_index].copy()
        # 修补过程中的MASK
        mask = masks[frame_index].copy()
        # 平移后的帧
        frame2s_shift = []
        for j in source:

            # cv2.imshow('inpaint', inpaint)
            # cv2.imshow('mask', mask)

            if j == 0 or masks[frame_index + j] is None:
                continue
            
            # 运动信息
            frame2_shift = alignImg(vertex, frames[frame_index], frames[frame_index + j], i, i + j)
            mask2_shift = alignImg(vertex, masks[frame_index], masks[frame_index + j], i, i + j)
            
            frame2s_shift.append(frame2_shift)
            # cv2.imshow('frame2_shift', frame2_shift)
            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()
            #     return

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
        
        mask = cv2.bitwise_and(mask, demask_3d)

        #---------------------用周围像素补洞-----------------------
        # inpaint = cv2.inpaint(inpaint, mask[..., 0], 5, cv2.INPAINT_TELEA)

        #----------------------中值滤波补洞-------------------------
        medianBlur = cv2.medianBlur(inpaint, 11)
        inpaint = np.where(mask > 128, medianBlur, inpaint)

        #----------------------帧平均补洞---------------------------
        # frame2s_shift = np.asarray(frame2s_shift)

        # # 找亮度最大的像素组
        # frame2s_shift_gray = []
        # for frame2_shift in frame2s_shift:
        #     frame2_shift_gray = cv2.cvtColor(frame2_shift, cv2.COLOR_BGR2GRAY)
        #     frame2s_shift_gray.append(frame2_shift_gray)
        # frame2s_shift_gray = np.asarray(frame2s_shift_gray)
        # max_index = np.argmax(frame2s_shift_gray, axis = 0)
        # max_index_3d = np.dstack((max_index, max_index, max_index))
        # frame_max = np.zeros_like(frame2s_shift[0])
        # for j in range(len(frame2s_shift)):
        #     frame_max = np.where(max_index_3d == j, frame2s_shift[j], frame_max)

        # # 去除亮度最大部分求平均
        # frame_sum = np.sum(frame2s_shift, axis = 0)
        # frame_avg = (frame_sum - frame_max) / (len(frame2s_shift) - 1)
        # frame_avg = frame_avg.astype('uint8')

        # # 补洞
        # inpaint = np.where(mask > 128, frame_avg, inpaint)

        #-----------------------恢复物体--------------------------
        inpaint = np.where(demask_3d == 0, frame, inpaint)
        
        #------------------------保存-----------------------------
        cv2.imwrite('dense/inpaint_dense/inpaint_{}.jpg'.format(i), inpaint)

        #------------------------展示-----------------------------
        if do_plot:
            cv2.imshow('mask_old', masks[frame_index])
            cv2.imshow('old', frames[frame_index])
            cv2.imshow('mask_new', mask)
            cv2.imshow('inpaint', inpaint)
            if cv2.waitKey(mode == 'run') & 0xff == ord('q'):
                break

        #-----------------------迭代------------------------------
        frames.pop(0)
        masks.pop(0)
        
        print(i, time() - start)

    if do_plot:
        cv2.destroyAllWindows()


# 多stride
strides = [3, 30]
stride_max = max(strides)
first_frame = 210
last_frame = 503

# 获取背景移动信息
vertex = findVertex('video.avi')
print('finsh')

makeMask(vertex, first_frame, last_frame, strides, mode = 'debug')
# paint(vertex, first_frame, last_frame, stride_max, mode = 'run')