import cv2

try:
    # Read the input image
    img = cv2.imread("CSC_0235.JPG")

    if img is None:
        raise Exception("Image missing.")

    # Apply Median Blur (kernel size = 5)
    median_out = cv2.medianBlur(img, 5)

    # Save output (same folder, same ratio, same resolution)
    cv2.imwrite("median_output.png", median_out)

    print("Saved as median_output.png")
    print("Input  Size (H x W):", img.shape[:2])
    print("Output Size (H x W):", median_out.shape[:2])

except Exception as e:
    print("Error:", e)
