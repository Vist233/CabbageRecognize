import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
image_path = "centered_22A-T-34-2球形侧视图.JPG"  # 替换为你的图片路径
image = cv2.imread(image_path)

# 将图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊来减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 查找图像中的轮廓
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图像上绘制轮廓
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# 显示结果
plt.figure(figsize=(10, 10))

# 显示原图
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 显示边缘检测后的图像
plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis('off')

# 显示绘制轮廓的图像
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title("Contours")
plt.axis('off')

plt.show()
