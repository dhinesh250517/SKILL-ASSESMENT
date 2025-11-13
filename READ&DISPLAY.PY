import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("DSC_0666.JPG")

if img is None:
    raise FileNotFoundError("Image not found.")

# Convert BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get height & width
h, w, _ = img.shape

# Display using exact pixel size
dpi = 100
plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

plt.imshow(img)
plt.axis("off")
plt.show()
