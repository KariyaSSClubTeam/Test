import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

img = cv2.imread("C:\\Users\\tuzuk\\Desktop\\point.JPG")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

contours =  cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 
def draw_contours(ax, img, contours):
    ax.imshow(img)
    ax.axis('off')
    for i, cnt in enumerate(contours):
        cnt = np.squeeze(cnt, axis=1)  # (NumPoints, 1, 2) -> (NumPoints, 2)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(Polygon(cnt, color='b', fill=None, lw=2))
        # 輪郭の点を描画する。
        ax.plot(cnt[:, 0], cnt[:, 1], 'ro', mew=0, ms=4)
      
fig, ax = plt.subplots(figsize=(6, 6))
draw_contours(ax, img, contours)
plt.show()