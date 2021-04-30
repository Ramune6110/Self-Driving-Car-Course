import cv2

image = cv2.imread('figure/lane_image.jpg')
cv2.imshow('result', image)
cv2.waitKey(0)
