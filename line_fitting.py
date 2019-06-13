import cv2
import numpy as np

img = cv2.imread("images/hand_gau_out1.JPG")

# 塗りつぶし
height, width, channels = img.shape[:3]

print("width: " + str(width))
print("height: " + str(height))

cv2.rectangle(img, (1, height), (width, 1), (0, 0, 0),
              thickness=100, lineType=cv2.LINE_4)

cv2.imwrite("images/hand_gau_draw.JPG", img)


# ガウスのぼかし処理
gau = cv2.GaussianBlur(img, (99, 99), 0)

# グレースケール化
gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

# ２値化
ret, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite("images/hand_gau_out2.JPG", th)

src = np.array(th, dtype="float32")

# cv2.imwrite("images/CIMG2478.JPG", src)

if len(src.shape) == 3:
    height, width, channels = src.shape[:3]
else:
    height, width = src.shape[:2]

    channels = 1

    print("dtype(src) " + str(src.dtype))


image, contours, hierarchy = cv2.findContours(
    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

print("dtype(cnt): " + str(cnt.dtype))

cnt_ = np.array(cnt, dtype="float32")

print("dtype(cnt_): " + str(cnt_.dtype))

rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img = cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 0), 2)


print(lefty)
print(righty)

cv2.imwrite("images/hand_gau_out1.JPG", img)
