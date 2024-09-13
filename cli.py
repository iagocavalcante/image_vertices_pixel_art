#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import requests
import sys
import os

def download_image(url):
    '''
    Downloads an image from a given URL and returns it as a NumPy array.
    :param url: URL to the image
    :return: image as a NumPy array
    '''
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download image from {url}")
        sys.exit(1)

    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    return img

def load_image(path_or_url):
    '''
    Loads an image from a local path or URL.
    :param path_or_url: file path or URL of the image
    :return: image as a NumPy array
    '''
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        print(f"Downloading image from {path_or_url}...")
        return download_image(path_or_url)
    elif os.path.exists(path_or_url):
        print(f"Loading image from {path_or_url}...")
        return cv2.imread(path_or_url, cv2.IMREAD_GRAYSCALE)
    else:
        print(f"Invalid path or URL: {path_or_url}")
        sys.exit(1)

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

def main(args):
    # Load image
    img = load_image(args.image)

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

    # Step 9: Save the image with drawn contours and yellow points
    output_image_path = args.output if args.output else "polygon_accurate_output.png"
    cv2.imwrite(output_image_path, im)

    # Step 10: Print the formatted vector points
    print("[")
    for vec in vector_points:
        print(f"  {vec},")
    print("]")
    print(f"Processed image saved to: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate polygon vertices from an image.")
    parser.add_argument("image", help="Path to the image or URL of the image.")
    parser.add_argument("--output", help="Path to save the output image with the drawn polygon.", default="polygon_accurate_output.png")

    args = parser.parse_args()
    main(args)
