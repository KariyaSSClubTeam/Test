import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

img = cv2.imread("C:\\Users\\tuzuk\\Desktop\\point.JPG")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

contours =  cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
result = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

cv2.imwrite("C:\\Users\\tuzuk\\Desktop\\point.JPG",result)