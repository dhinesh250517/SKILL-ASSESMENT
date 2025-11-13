

<div style="
    width: 100%;
    min-height: 1100px; 
    padding: 40px;
    box-sizing: border-box;
    border: 1px solid #000;
">

  <!-- Centered Image -->
  <p align="center">
    <img src="https://github.com/user-attachments/assets/a0944db9-92a4-48be-b312-ccf0248f766b" width="500">
  </p>

  <!-- Title -->
  <h2 align="center" style="margin-top: -10px;">SKILL ASSESSMENT I</h2>

  <br><br>

  <!-- Left-Aligned Details -->
  <p style="font-size: 18px; line-height: 1.8;">
    <b>Name:</b> DHINESH S <br>
    <b>Reg No:</b> 212223060053 <br>
    <b>Dept / Year:</b> ECE / III Year <br>
    <b>Course Code & Name:</b> 19EC503 – Digital Image Processing <br>
    <b>Slot:</b> 3U1-1
  </p>

  <!-- BLANK SPACE FOR PRINTING -->
  <div style="height: 600px;"></div>

</div>






















# Basic Image Processing:


## a.How do you read and display an image using OpenCV or PIL in Python? (5)
```
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("DSC_0666.JPG")
if img is None:
    raise FileNotFoundError("Image not found.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape
dpi = 100
plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.show()
```
### OUTPUT IMAGE:
<img src="https://github.com/user-attachments/assets/046ae47f-036f-44ab-8d5b-7804fe3daadf" width="200">

## b.Write a Python program to convert a color image to grayscale. (15)
```
import cv2
try:
    color_img = cv2.imread("sample_image.png")
    if color_img is None:
        raise Exception("Image not found. Check the filename.")
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original Image", color_img)
    cv2.imshow("Grayscale Image", gray_img)
    cv2.imwrite("grayscale_output.png", gray_img)
    print("Saved as grayscale_output.png (same size as input).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as err:
    print("Error:", err)

```
### OUTPUT IMAGE:
<img src="https://github.com/user-attachments/assets/60ed4181-14e2-4cea-bc78-ed5c20b03575" width="200">
<img src="https://github.com/user-attachments/assets/7175f734-1a9a-4ce9-8dc1-39540ab118b9" width="200">

# Image Filtering:
## a.How would you implement a Gaussian filter using Python's OpenCV library? (5)

```
import cv2

try:
    input_img = cv2.imread("DSC_0458.JPG")
    if input_img is None:
        raise Exception("Could not load image. Check filename.")
    blurred_img = cv2.GaussianBlur(input_img, (5, 5), 0)
    cv2.imwrite("gaussian_result.png", blurred_img)
    print("Saved as gaussian_result.png")
    print("Input size (H x W):", input_img.shape[0], "x", input_img.shape[1])
    print("Output size (H x W):", blurred_img.shape[0], "x", blurred_img.shape[1])
except Exception as err:
    print("Error:", err)
```
## OUTPUT:
<img src="https://github.com/user-attachments/assets/c11dd8ea-c3d4-49f3-8029-f19cce0618d9" width="200">

## b.Write a Python program to apply edge detection (Sobel or Canny) to an image. (15)
```

import cv2
try:
    raw_img = cv2.imread("DSC_0856.JPG", cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise Exception("Image not found during read operation.")
    smoothed = cv2.GaussianBlur(raw_img, (5, 5), 1.4)
    edges = cv2.Canny(smoothed, 100, 200)
    cv2.imwrite("edge_output.png", edges)
    print("Saved as edge_output.png")
    print("Input Size  (H x W):", raw_img.shape)
    print("Output Size (H x W):", edges.shape)
except Exception as e:
    print("Error:", e)

```

### OUTPUT :
<img src="https://github.com/user-attachments/assets/820d511a-250f-49d6-9e97-49bc4b591611" width="200">
<img src="https://github.com/user-attachments/assets/6f9706b6-dfed-43b8-add3-a1ebf1fe3296" width="200">

# Image Transformation:
## a.How can you rotate or resize an image using OpenCV in Python? (5)
```
import cv2
import numpy as np
try:
    img = cv2.imread("DSC_0580.JPG")
    if img is None:
        raise Exception("Image loading failed.")
    (h, w) = img.shape[:2]
    pivot = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(pivot, 180, 1.0)
    rotated_output = cv2.warpAffine(img, rot_matrix, (w, h))
    cv2.imwrite("rotated_180_output.png", rotated_output)
    print("Saved as rotated_180_output.png")
    print("Input Size  (H x W):", img.shape)
    print("Output Size (H x W):", rotated_output.shape)
except Exception as e:
    print("Error:", e)

```
### OUTPUT:
<img src="https://github.com/user-attachments/assets/2dc448f6-8c68-4f85-9361-2e1efdb3ef09" width="200">
<img src="https://github.com/user-attachments/assets/a791ddfb-4218-444b-b409-1cd916c34f49" width="200">

## b.Write a Python program to perform affine transformation on an image. (15)
```
import cv2
import numpy as np
try:
    img = cv2.imread("DSC_0476.JPG")
    if img is None:
        raise Exception("Image could not be opened.")
    src_pts = np.float32([[50, 50], [200, 60], [50, 200]])
    dst_pts = np.float32([[70, 120], [220, 90], [100, 260]])
    aff_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    transformed = cv2.warpAffine(img, aff_matrix, (img.shape[1], img.shape[0]))
    cv2.imwrite("affine_output.png", transformed)
    print("Saved as affine_output.png")
    print("Input  Size (H x W):", img.shape[:2])
    print("Output Size (H x W):", transformed.shape[:2])
except Exception as e:
    print("Error:", e)
```
### OUTPUT:
<img src="https://github.com/user-attachments/assets/4293cea2-9822-4eb1-9dc8-182c1d4f9cd3" width="200">
<img src="https://github.com/user-attachments/assets/c1b76946-d838-4401-9bf5-5ec46d18de0f" width="200">

# Image Enhancement:
## a.How can you adjust the brightness and contrast of an image using Python? (5)
```
import cv2
import os
try:
    img = cv2.imread("DSC_0490.JPG")
    if img is None:
        raise Exception("Image not found.")
    contrast = 1.5   # alpha
    brightness = 50  # beta
    enhanced = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    output_file = "bright_contrast_result.png"
    cv2.imwrite(output_file, enhanced)
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    if file_size > 10:
        print("PNG too large (", file_size, "MB ). Compressing...")
        os.remove(output_file)
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
```
### OUTPUT:
<img src="https://github.com/user-attachments/assets/56b539d1-a169-48b0-b528-6c3cbcd1c13a" width="200">
<img src="https://github.com/user-attachments/assets/27c4b603-3fba-448e-afdf-a7db23b386ba" width="200">

## b.Write a Python program to apply histogram equalization to an image. (15)

```
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
```
### OUTPUT:
<img src="https://github.com/user-attachments/assets/8bfd1349-f60b-4a2a-acef-eb3feec348d1" width="200">
<img src="https://github.com/user-attachments/assets/c2a7abd0-6cd8-4e68-9d56-7067327f9651" width="200">

# Noise Removal:
## How do you remove noise from an image using median filtering in Python? (5) 
```
import cv2
try:
    img = cv2.imread("sample_image.png")
    if img is None:
        raise Exception("Image missing.")
    median_out = cv2.medianBlur(img, 5)
    cv2.imwrite("median_output.png", median_out)
    print("Saved as median_output.png")
    print("Input  Size (H x W):", img.shape[:2])
    print("Output Size (H x W):", median_out.shape[:2])
except Exception as e:
    print("Error:", e)
```
### OUTPUT:

<img src="https://github.com/user-attachments/assets/9701feeb-ecfb-43cc-8da7-5b746e73fb09" width="200">
<img src="https://github.com/user-attachments/assets/a199992b-505c-4ba0-b933-2d2c9203146e" width="200">

## Write a Python program to apply bilateral filtering to an image. (15) 
```
import cv2
try:
    img = cv2.imread("DSC_0162.JPG")
    if img is None:
        raise Exception("Image could not be opened.")
    bilateral_out = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite("bilateral_output.png", bilateral_out)
    print("Saved as bilateral_output.png")
    print("Input  Size (H x W):", img.shape[:2])
    print("Output Size (H x W):", bilateral_out.shape[:2])
except Exception as e:
    print("Error:", e)

```
## OUTPUT:

<img src="https://github.com/user-attachments/assets/500fd417-84b1-4b89-a986-948003629e09" width="200">










    





