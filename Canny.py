import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


img = cv2.imread("C:\\Users\\tuzuk\\Desktop\\point.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
dst = cv2.Canny(gray,350,600)
     
cv2.imwrite("C:\\Users\\tuzuk\\Desktop\\point.JPG", dst)
