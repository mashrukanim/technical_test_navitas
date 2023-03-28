import cv2
import numpy as np

img = cv2.imread('1.jpg')
img = cv2.resize(img, (960,540))
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_threshold = np.array([40, 50, 50])
upper_threshold = np.array([80, 255, 255])

mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
# cv2.imshow('mask', mask)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
largest_contour_mask = np.zeros_like(mask)
# cv2.imshow('mask_largest', largest_contour_mask)
cv2.drawContours(largest_contour_mask, [largest_contour], 0, 255, -1)
result = cv2.bitwise_and(img, img, mask=largest_contour_mask)
outline = cv2.drawContours(img, [largest_contour], 0, (255,255,255),5)
cv2.imshow('Outline',outline)
# Display the image in the window
cv2.imshow('Output Window', result)
# Display the resulting image
# cv2.imshow('Green Object', green_img)
cv2.imwrite('Output_contour.jpg', outline)
cv2.imwrite('Output_result.jpg', result)
# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()