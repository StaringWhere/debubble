import cv2
import numpy as np
from pitFilter import pitFilter

cap = cv2.VideoCapture('video.avi')

# 定义一个VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1436,664))

count = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:
        # frame_filterd = pitFilter(frame)
        # out.write(frame_filterd
        # cv2.imshow('frame_filterd',frame_filterd)
        # cv2.imshow('frame', frame)
        cv2.imwrite('frames/frame_{}.jpg'.format(count),frame)
        count += 1
        # r = 100

        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    else:
        cap.release()
        # out.release()
        cv2.destroyAllWindows()

print(count)
