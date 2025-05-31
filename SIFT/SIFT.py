import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def generate_gaussian_pyramid(image, num_octaves=4, num_scales=5, sigma=1.6):
    k = 2 ** (1 / (num_scales - 3)) 
    pyramid = []

    for octave in range(num_octaves):
        scales = []
        for scale in range(num_scales):
            sigma_effective = sigma * (k ** scale)
            img_blur = gaussian_filter(image, sigma=sigma_effective)
            scales.append(img_blur)
        pyramid.append(scales)
        image = image[::2, ::2]
    return pyramid

def generate_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for scales in gaussian_pyramid:
        dogs = []
        for i in range(1, len(scales)):
            dogs.append(scales[i] - scales[i-1])
        dog_pyramid.append(dogs)
    return dog_pyramid

def detect_keypoints(dog_pyramid):
    keypoints = []
    for o, dogs in enumerate(dog_pyramid):
        for i in range(1, len(dogs) - 1):
            for y in range(1, dogs[i].shape[0] - 1):
                for x in range(1, dogs[i].shape[1] - 1):
                    patch = np.stack([dogs[i-1][y-1:y+2, x-1:x+2],
                                    dogs[i][y-1:y+2, x-1:x+2],
                                    dogs[i+1][y-1:y+2, x-1:x+2]])
                    val = dogs[i][y, x]
                    if val == patch.max() or val == patch.min():
                        keypoints.append((x * (2 ** o), y * (2 ** o), o, i))
    return keypoints

def assign_orientation(image, keypoints, radius=8, num_bins=36):
    oriented_keypoints = []
    for x, y, o, s in keypoints:
        mag, angle = compute_gradient_magnitude_orientation(image, x, y, radius)
        hist = np.zeros(num_bins)
        for m, a in zip(mag.flatten(), angle.flatten()):
            bin_idx = int(np.round(a * num_bins / 360)) % num_bins
            hist[bin_idx] += m
        max_bin = np.argmax(hist)
        max_val = hist[max_bin]
        for i in range(num_bins):
            if hist[i] >= 0.8 * max_val:
                theta = i * (360 / num_bins)
                oriented_keypoints.append((x, y, theta))
    return oriented_keypoints

def compute_gradient_magnitude_orientation(image, x, y, radius):
    patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return np.zeros_like(patch), np.zeros_like(patch)

    dx = patch[:, 2:] - patch[:, :-2]
    dy = patch[2:, :] - patch[:-2, :]

    dx = dx[1:-1, :]
    dy = dy[:, 1:-1]

    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    return magnitude, orientation


def compute_descriptors(image, keypoints, num_bins=8, window_size=16):
    descriptors = []
    for x, y, theta in keypoints:
        if x < 8 or y < 8 or x + 8 >= image.shape[1] or y + 8 >= image.shape[0]:
            continue
        desc = []
        patch = image[int(y)-8:int(y)+8, int(x)-8:int(x)+8]
        if patch.shape != (16, 16):
            continue
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                cell = patch[i:i+4, j:j+4]
                dx = cell[:, 2:] - cell[:, :-2]
                dy = cell[2:, :] - cell[:-2, :]
                dx = dx[1:-1, :]
                dy = dy[:, 1:-1]
                mag = np.sqrt(dx**2 + dy**2)
                ori = (np.degrees(np.arctan2(dy, dx)) - theta + 360) % 360
                hist = np.zeros(num_bins)
                for m, o in zip(mag.flatten(), ori.flatten()):
                    bin_idx = int(np.round(o * num_bins / 360)) % num_bins
                    hist[bin_idx] += m
                desc.extend(hist)
        desc = np.array(desc)
        desc /= np.linalg.norm(desc) + 1e-7
        desc = np.clip(desc, 0, 0.2)
        desc /= np.linalg.norm(desc) + 1e-7
        descriptors.append(desc)
    return np.array(descriptors)

orig_image1 = cv2.imread('Eiffel1.webp')
img1 = cv2.cvtColor(orig_image1, cv2.COLOR_BGR2GRAY)
img1 = np.float32(img1)
g_pyramid1 = generate_gaussian_pyramid(img1)
dog_pyramid1 = generate_dog_pyramid(g_pyramid1)
keypoints1 = detect_keypoints(dog_pyramid1)
oriented_kps1 = assign_orientation(img1, keypoints1)
descriptors1 = compute_descriptors(img1, oriented_kps1)


orig_image2 = cv2.imread('Eiffel2.webp')
img2 = cv2.cvtColor(orig_image2, cv2.COLOR_BGR2GRAY)
img2 = np.float32(img2)
g_pyramid2 = generate_gaussian_pyramid(img2)
dog_pyramid2 = generate_dog_pyramid(g_pyramid2)
keypoints2 = detect_keypoints(dog_pyramid2)
oriented_kps2 = assign_orientation(img2, keypoints2)
descriptors2 = compute_descriptors(img2, oriented_kps2)