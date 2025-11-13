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
## OUTPUT IMAGE:
<img src="https://github.com/user-attachments/assets/046ae47f-036f-44ab-8d5b-7804fe3daadf" width="200">


