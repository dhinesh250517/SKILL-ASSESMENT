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
'''
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

'''
### OUTPUT IMAGE:
<img src="https://github.com/user-attachments/assets/60ed4181-14e2-4cea-bc78-ed5c20b03575" width="200">
<img src="https://github.com/user-attachments/assets/7175f734-1a9a-4ce9-8dc1-39540ab118b9" width="200">





