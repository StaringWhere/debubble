import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt

#----------------------计算Movement--------------------
# 读取视频
cap = cv2.VideoCapture('video.avi')

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
ps = [p0[:,0]] # 保存角点

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

# 计算背景缩放和位移（先缩放后位移）
p0 = background_p[0]
background_scale = []
background_movement = []
for p1 in background_p[1: ]:
    vec0 = abs(p0[1] - p0[0])
    dis0 = np.sqrt(vec0[0] ** 2 + vec0[1] ** 2)
    vec1 = abs(p1[1] - p1[0])
    dis1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    scale = dis1 / dis0
    movement_x0 = p1[0][0] * scale - p0[0][0]
    movement_y0 = p1[0][1] * scale - p0[0][1]
    movement_x1 = p1[1][0] * scale - p0[1][0]
    movement_y1 = p1[1][1] * scale - p0[1][1]
    movement_x = (movement_x0 + movement_x1) / 2
    movement_y = (movement_y0 + movement_y1) / 2
    background_movement.append([movement_x, movement_y, scale])
    p0 = p1

#------------------------------------------------------
def calMultiMovement(movement, i, j):
    '''
    从第i帧到第j帧的运动，从第1帧开始计数
    '''
    movement_x = 0
    movement_y = 0
    scale = 1
    for k in np.arange(j - 2, i - 2, -1):
        movement_x += scale * movement[k][0]
        movement_y += scale * movement[k][1]
        scale *= movement[k][2]

    return movement_x, movement_y, scale

#--------------------画出scale曲线----------------------
scales = np.asarray(background_movement)[:, 2]

f = plt.figure(figsize = (12,3))
plt.rcParams['figure.dpi'] = 95
plt.rcParams['font.sans-serif'] = ['simhei']
plt.subplot(131)
plt.plot(scales)
plt.title('两帧之间的scale')

scales_since = []
scale_since = 1
for scale in scales:
    scales_since.append(scale_since)
    scale_since *= scale
plt.subplot(132)
plt.plot(scales_since)
plt.title('相对于第一帧的scale')

scales30 = []
for i in np.arange(1, 503 - 30 + 1, 1):
    scale30 = calMultiMovement(background_movement, i, i + 30)[2]
    scales30.append(scale30)
plt.subplot(133)
plt.plot(np.arange(30, 503, 1), scales30)
plt.title('相对于30帧前的scale')

plt.savefig('scales.jpg', dpi = 500)