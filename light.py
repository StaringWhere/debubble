import numpy as np
import cv2


# 读取视频
cap = cv2.VideoCapture('video.avi')

# ShiTomasi 角点检测参数
feature_params = dict( maxCorners = 100,
                    qualityLevel = 0.3,
                    minDistance = 7,
                    blockSize = 7 )

# lucas kanade光流法参数
lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机颜色
color = np.random.randint(0,255,(100,3))

# 获取第一帧，找到角点
ret, old_frame = cap.read()
#找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

indexs = [] # 选取的角点标号
movements = [] # 移动

while(1):
    ret,frame = cap.read()
    if type(frame) is type(None):
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

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
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

# 计算位移
background_movement = []
for i, movement in enumerate(movements):
    for j in indexs[i: ]:
        movement = movement[j]
    movement = np.mean(movement[0], axis = 0)
    background_movement.append(movement)