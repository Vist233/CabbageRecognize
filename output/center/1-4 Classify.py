import cv2
import numpy as np

# 读取图像
image = cv2.imread('centered_22A-T-34-2球形侧视图.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 获取图像尺寸
height, width = gray.shape

# 计算图像的上面三分之一的高度
top_third_height = height // 3

# 二值化处理
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# 轮廓检测
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像，用于显示结果
contour_image = np.zeros_like(gray)

# 遍历每个轮廓，保留在上面三分之一区域内的轮廓
for cnt in contours:
    # 检查轮廓的所有点是否位于上面三分之一的区域
    if any(point[0][1] < top_third_height for point in cnt):
        # 绘制轮廓
        cv2.drawContours(contour_image, [cnt], -1, (255), 2)  # 用白色绘制

# 显示结果
cv2.imshow('Top Third Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
