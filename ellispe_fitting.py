import cv2
import numpy as np

# 画像の読み込み
img = cv2.imread("images/volto_copy.png")


# 塗りつぶし
height, width, channels = img.shape[:3]

print("width: " + str(width))
print("height: " + str(height))

cv2.rectangle(img, (1, height), (width, 1), (0, 0, 0),
              thickness=30, lineType=cv2.LINE_4)

cv2.imwrite("images/CIMG4192.JPG", img)


# ガウスのぼかし処理
gau = cv2.GaussianBlur(img, (5, 5), 0)

# グレースケール化
gray = cv2.cvtColor(gau, cv2.COLOR_BGR2GRAY)

# ２値化
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("images/volto_out2.png", th)

#
src = np.array(th, dtype="float32")


if len(src.shape) == 3:
    height, width, channels = src.shape[:3]
else:
    height, width = src.shape[:2]

    channels = 1

    print("dtype(src) " + str(src.dtype))

# 輪郭抽出
image, contours, hierarchy = cv2.findContours(
    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

print("dtype(cnt): " + str(cnt.dtype))

cnt_ = np.array(cnt, dtype="float32")

print("dtype(cnt_): " + str(cnt_.dtype))

# 楕円フィッティング
ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

# 焦点
print(ellipse)

cv2.imwrite("images/volto_out1.png", img)

print("OK")
