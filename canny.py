import cv2

img = cv2.imread("images/point.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
dst = cv2.Canny(gray, 350, 600)

cv2.imwrite("images/point.JPG", dst)
