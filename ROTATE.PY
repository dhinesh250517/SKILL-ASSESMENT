import cv2
import numpy as np

try:
    # Read input image
    img = cv2.imread("DSC_0580.JPG")

    if img is None:
        raise Exception("Image loading failed.")

    # Get original size
    (h, w) = img.shape[:2]
    pivot = (w // 2, h // 2)

    # Rotation matrix for 180 degrees
    rot_matrix = cv2.getRotationMatrix2D(pivot, 180, 1.0)

    # Rotate image (same size)
    rotated_output = cv2.warpAffine(img, rot_matrix, (w, h))

    # Save output
    cv2.imwrite("rotated_180_output.png", rotated_output)

    print("Saved as rotated_180_output.png")
    print("Input Size  (H x W):", img.shape)
    print("Output Size (H x W):", rotated_output.shape)

except Exception as e:
    print("Error:", e)
