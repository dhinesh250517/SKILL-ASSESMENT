import cv2
import numpy as np
try:
    gray_img = cv2.imread(".trashed-1754155602-DSC_0331.JPG", cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        raise Exception("Image could not be read.")
    eq_img = cv2.equalizeHist(gray_img)
    cv2.imwrite("hist_eq_output.png", eq_img)
    print("Saved as hist_eq_output.png")
    print("Input Size  (H x W):", gray_img.shape)
    print("Output Size (H x W):", eq_img.shape)
except Exception as e:
    print("Error:", e)
