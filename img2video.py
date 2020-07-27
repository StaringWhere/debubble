import cv2
import numpy as np
import os

# 定义一个VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1436,664 * 2))

src = 'dense/inpaint_dense/'
n = len(os.listdir(src))

with_original_frame = 0

for i in range(n):
    inpaint = cv2.imread(src + 'inpaint_{}'.format(i + 1) + '.jpg')
    if with_original_frame:
        frame = cv2.imread('frames/frame_{}'.format(i + 1) + '.jpg')
    if inpaint is not None:
        if with_original_frame:
            img = np.vstack((frame, inpaint))
            img = img.astype('uint8')
            out.write(img)
            img_resize = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
            cv2.imshow('img', img_resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out.write(inpaint)
            cv2.imshow('inpaint', inpaint)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

print(i)
out.release()
cv2.destroyAllWindows()
