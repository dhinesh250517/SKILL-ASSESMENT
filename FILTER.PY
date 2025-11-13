import cv2

try:
    # Load the input image
    input_img = cv2.imread("DSC_0458.JPG")

    if input_img is None:
        raise Exception("Could not load image. Check filename.")

    # Apply Gaussian Blur (DOES NOT change size or ratio)
    blurred_img = cv2.GaussianBlur(input_img, (5, 5), 0)

    # Save output (same folder, same ratio, same size)
    cv2.imwrite("gaussian_result.png", blurred_img)

    print("Saved as gaussian_result.png")
    print("Input size (H x W):", input_img.shape[0], "x", input_img.shape[1])
    print("Output size (H x W):", blurred_img.shape[0], "x", blurred_img.shape[1])

except Exception as err:
    print("Error:", err)
