import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = "centered_22A-T-34-2球形侧视图.JPG"
image = cv2.imread(image_path)

# 转换为HSV颜色空间，便于处理颜色差异
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义绿色的HSV范围，用于分割白菜叶子
lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 255, 255])

# 创建掩码，识别绿色区域
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 对掩码进行形态学处理，去除噪点并平滑叶片边缘
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 进行边缘检测
edges = cv2.Canny(mask, 100, 200)

# 显示结果
plt.figure(figsize=(10, 10))

# 原始图像
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 掩码图像
plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask of Green Areas')
plt.axis('off')

# 边缘检测图像
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edges of Leaves')
plt.axis('off')

plt.show()
