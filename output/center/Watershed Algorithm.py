import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
image_path = "centered_22A-T-34-2球形侧视图.JPG"  # 替换为你的图片路径
image = cv2.imread(image_path)

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊来减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 应用Otsu阈值分割
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 进行形态学操作去除噪声
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 找到图像的背景
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 通过距离变换找出前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 找到未知区域（背景和前景之间的重叠区域）
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记标签
_, markers = cv2.connectedComponents(sure_fg)

# 增加标签区域，以确保背景的标签为1
markers = markers + 1

# 将未知区域标记为0
markers[unknown == 255] = 0

# 使用分水岭算法
markers = cv2.watershed(image, markers)

# 将边界标记为红色
image[markers == -1] = [255, 0, 0]

# 显示结果
plt.figure(figsize=(10, 10))

# 显示原图
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image with Watershed Segmentation")
plt.axis('off')

# 显示阈值图像
plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')

# 显示分水岭标记
plt.subplot(1, 3, 3)
plt.imshow(markers, cmap='jet')
plt.title("Watershed Markers")
plt.axis('off')

plt.show()
