import cv2
import numpy as np

#Reading, resizing and changing the image format to HSV
img = cv2.imread('1.jpg')
img = cv2.resize(img, (960,540))
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Setting the threshold values for green color
lower_threshold = np.array([40, 50, 50])
upper_threshold = np.array([80, 255, 255])

#Calculating the mask/Region of interest
mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
# cv2.imshow('mask', mask)

#Finding the contours in the mask and identifying the largest contour
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

#Creating a new mask with only the largest contour
largest_contour_mask = np.zeros_like(mask)
largest_contour_mask = cv2.drawContours(largest_contour_mask, [largest_contour], 0, 255, -1)

#Calculating the maximum and minimum coordinates of the largest contour for drawing bounding box
xmin, ymin, xmax, ymax = float('inf'), float('inf'), 0, 0
for coords in largest_contour:
    x, y = coords[0]
    if x < xmin:
        xmin = x
    if y < ymin:
        ymin = y
    if x > xmax:
        xmax = x
    if y > ymax:
        ymax = y

#Drawing the bounding box
b_box = img.copy()
b_box = cv2.rectangle(b_box, (xmin, ymin), (xmax, ymax), (0,0,255), 2)

#Drawing the outline of the laptop screen
outline = cv2.drawContours(img, [largest_contour], 0, (0,0,255), 2)

#Extracting the green screen portion using the new mask
result = cv2.bitwise_and(img, img, mask=largest_contour_mask)

#Displaying the results
cv2.imshow('bbox', b_box)
cv2.imshow('Outline',outline)
cv2.imshow('Output Window', result)

#Generating output files
cv2.imwrite('Output_contour.jpg', outline)
cv2.imwrite('Output_result.jpg', result)
cv2.imwrite('Bounding_box.jpg',b_box)

cv2.waitKey(0)

cv2.destroyAllWindows()