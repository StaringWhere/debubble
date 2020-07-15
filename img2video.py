import cv2
import os

# 定义一个VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1436,664))

src = 'dense/inpaint_dense/'
n = len(os.listdir(src))

for i in range(n):
    frame = cv2.imread(src + 'inpaint_{}'.format(i + 1) + '.jpg')
    if frame is not None:
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        out.release()
        cv2.destroyAllWindows()
