import cv2
import numpy as np
import matplotlib.pyplot as plt

orig_image = cv2.imread('harris_image_1.webp')
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
image = np.float32(image)

Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

A = cv2.GaussianBlur(Ix*Ix, (5,5), 1)
B = cv2.GaussianBlur(Iy*Iy, (5,5), 1)
C = cv2.GaussianBlur(Ix*Iy, (5,5), 1)

k = 0.04
R = (A * B - C ** 2) - k * (A + B) ** 2

threshold = 0.01 * R.max()
corner_image = orig_image.copy()
corner_image[R > threshold] = [0, 0, 255]

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners (Manual)')
plt.axis('off')
plt.show()