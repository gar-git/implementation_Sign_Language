import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from skimage import io
from skimage.filters import median


img_salt_pepper_noise = cv2.imread('0.jpg', 0)
img = img_salt_pepper_noise
median_using_cv2 = cv2.medianBlur(img, 3)
from skimage.morphology import disk
median_using_skimage = median(img, disk(3), mode='constant', cval=0.0)
cv2.imshow("Original", img)
cv2.imshow("Using skimage median", median_using_skimage)
cv2.imwrite("median.jpg", median_using_skimage)

cv2.waitKey(0)
cv2.destroyAllWindows()