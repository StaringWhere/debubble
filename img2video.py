import cv2
import numpy as np
import os

#-------------------PARAMETERS--------------------
# 是否拼接
mode = 1
# 输出文件名（无后缀名）
filename = 'output'
# 文件夹1
dir1 = 'inpaints/inpaint_optic_v3.3'
# 文件夹2
dir2 = 'inpaints/inpaint_optic_v2.2'

#-------------------------------------------------
# 定义一个VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename + '.mp4',fourcc, 20.0, (1436,664 * 2))

# 文件序号正则表达式
proc = re.compile('_\d+\.')
rule = lambda str: int(proc.search(str)[0][1: -1])

# 文件夹1
src1 = os.listdir(dir1)
src1.sort(key = rule)

if mode:
    # -----------上下拼接--------------

    # 文件夹2
    src2 = os.listdir(dir2)
    src2.sort(key = rule)

    # 长度
    n = len(src1) if len(src1) <= len(src2) else len(src2)

    # 截取一样长的
    src1 = src1[: n]
    src2 = src2[: n]

    for s1, s2 in zip(src1, src2):
        img1 = cv2.imread(os.path.join(dir1, s1))
        img2 = cv2.imread(os.path.join(dir2, s2))

        img = np.vstack((img1, img2))
        img = img.astype('uint8')
        out.write(img)
        img_resize = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
        cv2.imshow('img', img_resize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    n = len(src1)
    for s1 in src1:
        img = cv2.imread(os.path.join(dir1, s1))
        out.write(img)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print('Total of', n, 'frames')
out.release()
cv2.destroyAllWindows()
