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


