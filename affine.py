import cv2
import numpy as np

try:
    # Load input image
    img = cv2.imread("DSC_0476.JPG")

    if img is None:
        raise Exception("Image could not be opened.")

    # --- Source & Destination Points (Same as your Colab code) ---
    src_pts = np.float32([[50, 50], [200, 60], [50, 200]])
    dst_pts = np.float32([[70, 120], [220, 90], [100, 260]])

    # Affine Transformation Matrix
    aff_matrix = cv2.getAffineTransform(src_pts, dst_pts)

    # Apply transformation (same size as input)
    transformed = cv2.warpAffine(img, aff_matrix, (img.shape[1], img.shape[0]))

    # Save result
    cv2.imwrite("affine_output.png", transformed)

    print("Saved as affine_output.png")
    print("Input  Size (H x W):", img.shape[:2])
    print("Output Size (H x W):", transformed.shape[:2])

except Exception as e:
    print("Error:", e)
