import cv2

try:
    # Load input image (same size & ratio preserved)
    img = cv2.imread("DSC_0162.JPG")

    if img is None:
        raise Exception("Image could not be opened.")

    # Apply Bilateral Filter
    bilateral_out = cv2.bilateralFilter(img, 9, 75, 75)

    # Save output automatically
    cv2.imwrite("bilateral_output.png", bilateral_out)

    print("Saved as bilateral_output.png")
    print("Input  Size (H x W):", img.shape[:2])
    print("Output Size (H x W):", bilateral_out.shape[:2])

except Exception as e:
    print("Error:", e)
