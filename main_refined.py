#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# Load image as a grayscale
img = cv2.imread("idle1.png", 0)


def fillhole(input_image):
    '''
    Fill holes in a binary image using flood fill.
    :param input_image: grayscale binary image
    :return: image with filled holes
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out

# Step 1: Fill the holes in the binary image
res = fillhole(img)

# Step 2: Find contours in the processed image
contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 3: Find the contour with the largest area (to match the yellow contour)
contour = max(contours, key=cv2.contourArea)

# Step 4: Approximate the contour with a lower epsilon for a tighter fit
# Lower epsilon for higher precision
peri = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.005 * peri, True)  # Lowered to 0.005 for more points

# Step 5: Create an output image (convert grayscale to BGR)
im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Step 6: Define the size of the square marker for each point (for visualization)
s = 5  # Slightly smaller marker size for accuracy

# Step 7: Prepare the result as Vector2 format and draw the points on the image
vector_points = []

for p in approx:
    p = p[0]  # Get the point from the approxPolyDP result
    vector_points.append(f"Vector2({p[0]}, {p[1]})")
    # Draw a small yellow square around each point for visualization
    im[p[1]-s:p[1]+s, p[0]-s:p[0]+s] = (0, 255, 255)  # Yellow color for points

# Step 8: Draw the contour in a different color for visualization (cyan)
cv2.drawContours(im, [approx], -1, (255, 255, 0), 2)

# Step 9: Show the image with contour and yellow points
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", im)
cv2.waitKey(0)

# Step 10: Save the image with drawn contours and yellow points
cv2.imwrite("polygon_accurate_output.png", im)

# Step 11: Print the formatted vector points
print("[")
for vec in vector_points:
    print(f"  {vec},")
print("]")
