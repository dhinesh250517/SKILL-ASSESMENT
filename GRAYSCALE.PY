import cv2

try:
    # Load input image
    color_img = cv2.imread("DSC_0747.JPG")

    if color_img is None:
        raise Exception("Image not found. Check the filename.")

    # Convert to grayscale (same size automatically)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Show images (they appear original size)
    cv2.imshow("Original Image", color_img)
    cv2.imshow("Grayscale Image", gray_img)

    # Save with same size
    cv2.imwrite("grayscale_output.png", gray_img)

    print("Saved as grayscale_output.png (same size as input).")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as err:
    print("Error:", err)
