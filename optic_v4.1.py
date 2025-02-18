
# %%
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt

'''
=====================光流法v4.1========================
optic v4.1

相较optic v3.3更新：
1. 减少不必要的IO

====================================================
'''

# %%
def findVertex(frames, do_show = 0):
    '''
    利用orb找特征点，利用稀疏光流跟踪找出背景角点，背景角点的特
    征的是移动缓慢，当角点速度小于阈值的时
    间大于200帧时即视为背景角点，并允许中间
    帧加入的角点，速度较慢。

    :param src  : 视频地址
    :type src   : String
    :param do_show  : 是否将过程展示出来，默认为不展示
    :returns    : 留存时间大于200的角点，
                  格式为(frame_num, p_num, 2)
                  若当前帧没有该角点，
                  则补充[-1, -1]
    '''

    start = time()

    # lucas kanade光流法参数
    lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 创建随机颜色
    color = np.random.randint(0, 255, (3000, 3))

    # 找到原始灰度图
    old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    old_gray = sharpen(old_gray)
    old_gray = cv2.equalizeHist(old_gray)

    p0 = orbFeature(frames[0])

    # 创建一个蒙版用来画轨迹
    mask = np.zeros_like(frames[0])

    flags = [np.ones(len(p0))]  # 选取的角点标号
    ps = [p0[:,0]] # 保存角点
    out_indexs = np.array([]) # 中途超速的点索引

    for frame in frames[1: ]:
        
        if frame is None:
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
        flag = np.zeros_like(flags[-1])
        flag[np.where(flags[-1] == 1)[0][index.flatten()]] = 1

        good_new = p1[index]
        good_old = p0[index]

        # 若有点速度大于阈值，则将其过去痕迹也抹去
        out_index = np.where(flags[-1] == 1)[0][((st == 1) & (mag >= 5)).flatten()]
        out_indexs = np.append(out_indexs, out_index)

        # 展示
        if do_show:
            
            # 画出轨迹
            frame_circle = frame.copy()
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame_circle = cv2.circle(frame_circle, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame_circle, mask)

            cv2.imshow('frame', img)
            # cv2.imshow('frame_gray', frame_gray)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        
        # 更新上一帧的图像和追踪点
        old_gray = frame_gray.copy()
        p_add = orbFeature(frame)
        p0 = good_new.reshape(-1, 1, 2)
        p0, add_num = appendVertex(p0, p_add, 20 / 1.414)
        flag = np.append(flag, np.ones(add_num))
        flags.append(flag.astype('int'))
        ps.append(p0[:, 0])

    if do_show:
        cv2.destroyAllWindows()

    # -----------将留存时间不够长的的点去掉------------

    # 最小留存时间
    min_duration = 50

    # 统计每个点的留存时间
    duration = np.zeros(flags[-1].shape)
    for flag in flags:
        duration[np.where(flag == 1)[0]] += 1

    # 将有效角点对应起来，若当前帧没有该角点，补充为[-1, -1]
    valid_index = np.where(duration > min_duration)[0]
    valid_index = np.array([index for index in valid_index if index not in out_indexs])

    bg_ps = []
    junk_p = np.array([-1, -1], dtype = 'int')
    for flag, p in zip(flags, ps):
        bg_p = []
        for index in valid_index:
            try:
                if flag[index] == 0:
                    bg_p.append(junk_p)
                else:
                    bg_p.append(p[int(sum(flag[: index]))])
            except:
                bg_p.append(junk_p)

        bg_ps.append(bg_p)

    bg_ps = np.asarray(bg_ps)
    print('vertex find complete in ', time() - start)
    return bg_ps

#%%
def orbFeature(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 0.03s
    gray = cv2.medianBlur(gray, 21)
    gray = cv2.equalizeHist(gray)

    # 0.015s
    orb = cv2.ORB_create(nfeatures = 100, scaleFactor = 2, firstLevel = 0)
    keypoints, _ = orb.detectAndCompute(gray, None)

    p = np.array([keypoint.pt for keypoint in keypoints]).reshape(-1, 1, 2)

    return p.astype('float32')


def appendVertex(p0, p_add, min_dis):
    '''
    根据4邻域距离加入新的角点
    '''
    p_ret = p0.copy()
    for p in p_add:
        vec = p0 - p
        dis = abs(vec[:, 0, 0]) + abs(vec[:, 0, 1])
        if all(dis > min_dis):
            p_ret = np.vstack((p_ret, p.reshape(-1, 1, 2)))
    
    add_num = p_ret.shape[0] - p0.shape[0]

    return p_ret, add_num


def sharpen(img):
    '''
    锐化图像
    '''
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    sharpened = cv2.filter2D(img, -1, kernel=kernel)

    return sharpened


def alignImg(vertex, ref, img, id_ref, id_img, fill = 1, p_th = 30):
    # 可用特征点
    flag = (np.sum(vertex[id_img - 1], axis = 1) >= 0) & (np.sum(vertex[id_ref - 1], axis = 1) >= 0)
    p_th = sum(flag) if sum(flag) < p_th else p_th
    p_img = vertex[id_img - 1][flag]
    p_ref = vertex[id_ref - 1][flag]

    # 计算平均移动距离
    vec = np.mean(np.abs(p_img - p_ref), axis = 0)
    mag = np.sqrt(vec[0] ** 2 + vec[1] ** 2)

    # 从外沿向内取一定数量的点
    p_num = 0
    p_ref_conv = np.empty((0, 2))
    p_img_conv = np.empty((0, 2))

    while p_num < p_th:
        flag_conv = cv2.convexHull(p_ref.reshape(1, -1, 2).astype('int'), returnPoints = False).flatten()
        p_num += len(flag_conv)

        p_ref_conv = np.append(p_ref_conv, p_ref[flag_conv], axis = 0)
        p_img_conv = np.append(p_img_conv, p_img[flag_conv], axis = 0)

        deflag_conv = [i not in flag_conv for i in range(len(p_ref))]
        p_ref = p_ref[deflag_conv]
        p_img = p_img[deflag_conv]

    # 计算单应性矩阵
    # h, mask = cv2.findHomography(p_img_conv, p_ref_conv, cv2.RANSAC, ransacReprojThreshold = mag)
    h, mask = cv2.findHomography(p_img_conv, p_ref_conv, method = 0)
    # 变换
    height, width = img.shape[: 2]

    if fill:
        img_reg = cv2.warpPerspective(img, h, (width, height), flags = cv2.INTER_NEAREST)
        ref_mb = cv2.medianBlur(ref, 21)
        img_reg = np.where(img_reg == 0, ref_mb, img_reg)

    else:
        try:
            channel = img.shape[2]
        except:
            channel = 1
        img_reg = cv2.warpPerspective(img, h, (width, height), flags = cv2.INTER_NEAREST, borderValue = [255] * channel)

    return img_reg, mag


def makeVideo(imgs, filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename + '.mp4',fourcc, 20.0, (1436,664))

    for img in imgs:
        out.write(img)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

# %%
def makeMask(frames, vertex, first_frame, last_frame, strides, do_plot = 1, mode = 'run'):
    height, width, channel = frames[0].shape

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

    all_masks = []

    for i in np.arange(first_frame, last_frame - stride_max + 1, 1):

        start = time()

        # --------------------读取图片----------------------
        frame = frames[i - 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        masks = []
        for stride in strides:
            frame2 = frames[i + stride - 1]
            if frame2 is None:
                break
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            #--------------------恢复背景位移-------------------
            gray2_shift, mag_bg = alignImg(vertex, gray, gray2, i, i + stride)

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
            # mask = mag > 5 + stride * 0.05
            mask = mag > 0.6 * mag_bg + 1.8
            mask = np.where(mag < 2, 0, mask)
            mask = (mask * 255).astype('uint8')

            masks.append(mask)

        mask = np.zeros((height, width)).astype('uint8')
        for mask2 in masks:
            mask = cv2.bitwise_or(mask, mask2)

        # --------------------hsv过滤-----------------------
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ret, mask1 = cv2.threshold(frame_hsv[..., 2], 100, 255, cv2.THRESH_BINARY)
        ret, mask2 = cv2.threshold(frame_hsv[..., 1], 80, 255, cv2.THRESH_BINARY_INV)
        mask_hsv = cv2.bitwise_and(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_hsv = cv2.dilate(mask_hsv, kernel, iterations = 1)

        mask = cv2.bitwise_or(mask_hsv, mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations = 1)
        mask = cv2.erode(mask, kernel, iterations = 1)



        #-----------------------展示-----------------------------
        if do_plot:
            cv2.imshow('frame2', gray2_shift)
            cv2.moveWindow("frame2",0,0)
            cv2.imshow('rgb', rgb)
            cv2.moveWindow("rgb",0,0)
            cv2.imshow('frame', frame)
            cv2.moveWindow("frame",0,0)
            cv2.imshow('mask', mask)
            cv2.moveWindow("mask",0,0)
            # cv2.imshow('img_inpaint', img_inpaint)

            if cv2.waitKey(mode == 'run') & 0xff == ord('q'):
                break

        #----------------------保存------------------------------
        # cv2.imwrite('dense/masks_dense/mask_{}.jpg'.format(i), mask)
        # cv2.imwrite('dense/rgb_dense/rgb_{}.jpg'.format(i), rgb)

        all_masks.append(mask)
        print(i, time() - start)

    if do_plot:
        cv2.destroyAllWindows()

    return all_masks


# %%
def paint(all_masks, all_frames, vertex, first_frame, last_frame, stride, do_plot = 1, mode = 'run'):
    all_inpaints = []
    frames = []
    masks = []
    source = np.arange(-10, 11, 1) # 从其他图片补第0张图片，必须为间距是1的等差数列
    frame_index = list(source).index(0) # 待修复的帧在数组中的位置
    for i in np.arange(first_frame, last_frame - stride + 1, 1):
        start = time()

        #---------------------读取---------------------
        if i == first_frame:
            for j in source[:-1]:
                try:
                    frames.append(all_frames[i + j - 1])
                    masks.append(all_masks[i + j -1])
                except IndexError:
                    frames.append(None)
                    masks.append(None)
        
        try:
            frames.append(all_frames[i + source[-1] - 1])
            masks.append(all_masks[i + source[-1] - 1])
        except IndexError:
            frames.append(None)
            masks.append(None)

        #-----------------------用前后帧补洞----------------------
        # 修补的图像
        inpaint = frames[frame_index].copy()
        # 修补过程中的MASK
        mask = masks[frame_index].copy()
        # 平移后的帧
        frame2s_shift = []
        for j in source[np.argsort(np.abs(source))]:

            # cv2.imshow('inpaint', inpaint)
            # cv2.imshow('mask', mask)
            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()
            #     return

            if j == 0 or masks[frame_index + j] is None:
                continue
            
            # 运动信息
            frame2_shift, _ = alignImg(vertex, frames[frame_index], frames[frame_index + j], i, i + j, fill = 0)
            mask2_shift, _ = alignImg(vertex, masks[frame_index], masks[frame_index + j], i, i + j, fill = 0)
            
            # print(i + j)
            # frame2s_shift.append(frame2_shift)
            # cv2.imshow('frame2_shift', frame2_shift)

            replace_mask = cv2.subtract(mask, mask2_shift)
            inpaint = np.where(np.dstack((replace_mask, replace_mask, replace_mask)) > 128, frame2_shift, inpaint)
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
        
        mask = cv2.bitwise_and(mask, demask)

        #---------------------用周围像素补洞-----------------------
        # inpaint = cv2.inpaint(inpaint, mask[..., 0], 5, cv2.INPAINT_TELEA)

        #----------------------中值滤波补洞-------------------------
        medianBlur = cv2.medianBlur(inpaint, 11)
        inpaint = np.where(np.dstack((mask, mask, mask)) > 128, medianBlur, inpaint)

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
        inpaint = np.where(np.dstack((demask, demask, demask)) == 0, frame, inpaint)
        
        #------------------------保存-----------------------------
        # cv2.imwrite('dense/inpaint_dense/inpaint_{}.jpg'.format(i), inpaint)

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
        
        all_inpaints.append(inpaint)

        print(i, time() - start)

    if do_plot:
        cv2.destroyAllWindows()
    
    return(all_inpaints)

#%%
class videoReader():
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)

    def release(self):
        self.cap.release()

    def readFrames(self, num):
        frames = []
        for i in range(num):
            ret, frame = self.cap.read()
            if frame is None:
                print('It is the last frame')
                break
            frames.append(frame)
        
        return frames

# %%

cap = videoReader('video.avi')
frames = cap.readFrames(1000)
cap.release()

# 多stride
strides = [3, 20]
stride_max = max(strides)
first_frame = 1
last_frame = 503

# 获取背景移动信息
vertex = findVertex(frames, do_show = 0)

# vertex = np.load('vertex.npy')

all_masks = makeMask(frames, vertex, first_frame, last_frame, strides, mode = 'run', do_plot = 0)
all_inpaints = paint(all_masks, frames, vertex, first_frame, last_frame, stride_max, mode = 'run', do_plot = 0)
makeVideo(all_inpaints, 'video')

# %%
