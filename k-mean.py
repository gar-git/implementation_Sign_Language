import numpy as np
import cv2

img = cv2.imread("median.jpg")


# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))

img2 = np.float32(img2)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Number of clusters
k = 4

attempts = 10

ret,label,center=cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)


res = center[label.flatten()]
res2 = res.reshape((img.shape)) 
cv2.imwrite("segmented.jpg", res2)


"""
#Now let us visualize the output result
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img2)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(res2)
plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()
"""