import cv2
import os

try:
    img = cv2.imread("DSC_0490.JPG")

    if img is None:
        raise Exception("Image not found.")

    # ------ Brightness & Contrast Adjustment ------
    contrast = 1.5   # alpha
    brightness = 50  # beta
    enhanced = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # ------ Try saving PNG first ------
    output_file = "bright_contrast_result.png"
    cv2.imwrite(output_file, enhanced)

    # Check file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB

    # ------ If file > 10MB -> compress using JPEG ------
    if file_size > 10:
        print("PNG too large (", file_size, "MB ). Compressing...")

        # Delete large PNG
        os.remove(output_file)

        # Start with high quality and reduce until <10MB
        quality = 95
        output_file = "bright_contrast_result.jpg"

        while True:
            cv2.imwrite(output_file, enhanced, [cv2.IMWRITE_JPEG_QUALITY, quality])
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB

            if file_size <= 10 or quality <= 10:
                break

            quality -= 5  # reduce quality gradually

        print(f"Compressed to {file_size:.2f} MB at quality={quality}")

    else:
        print(f"Saved PNG under 10MB ({file_size:.2f} MB)")

    print("Final saved file:", output_file)

except Exception as e:
    print("Error:", e)
