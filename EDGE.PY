import cv2

try:
    # Read input image in grayscale (same size & ratio)
    raw_img = cv2.imread("DSC_0856.JPG", cv2.IMREAD_GRAYSCALE)

    if raw_img is None:
        raise Exception("Image not found during read operation.")

    # Apply Gaussian smoothing (does NOT change size)
    smoothed = cv2.GaussianBlur(raw_img, (5, 5), 1.4)

    # Canny edge detection (same size & ratio)
    edges = cv2.Canny(smoothed, 100, 200)

    # Save output automatically in same folder
    cv2.imwrite("edge_output.png", edges)

    # Print sizes for confirmation
    print("Saved as edge_output.png")
    print("Input Size  (H x W):", raw_img.shape)
    print("Output Size (H x W):", edges.shape)

except Exception as e:
    print("Error:", e)
