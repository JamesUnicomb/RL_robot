import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/home/james/ROSPackages/rl_robot/src/rl_robot/rl_path_follow/data/rosbag_data/images/frame00121.jpg')

rows,cols,ch = img.shape

pts = [[[490,320],[640,480],[0,480],[150,320]],
        [[640,0],[640,480],[0,480],[0,0]]]    

M = cv2.getPerspectiveTransform(np.float32(pts[0]), np.float32(pts[1]))

dst = cv2.warpPerspective(img,M,(cols,rows))

dst = cv2.resize(dst, (0,0), fx=0.125, fy=0.125)

cv2.polylines(img, [np.array(pts[0])], 1, (255, 0, 0), thickness=5)

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')

plt.show()
