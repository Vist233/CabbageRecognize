import cv2
import numpy as np

def calculate_image_metrics(image_path):
    # 读取输入图片
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取图片的尺寸
    height, width = image.shape[:2]

    # 计算纵轴宽度除以横轴宽度的比值
    aspect_ratio = height / width

    # 定义非黑色像素的阈值
    non_black_threshold = 10

    # 计算每个部分的高度
    quarter_height = height // 4

    # 计算每个部分中非黑色像素的个数
    non_black_counts = []
    for i in range(1, 4):
        section = image[(i-1)*quarter_height:i*quarter_height, :]
        non_black_count = np.sum(np.any(section > non_black_threshold, axis=2))
        non_black_counts.append(non_black_count)
    sum = non_black_counts[0] + non_black_counts[1] + non_black_counts[2]
    uppRa = non_black_counts[0] / sum 
    midRa = non_black_counts[1] / sum
    lowRa = non_black_counts[2] / sum
    if midRa > 0.6:
        if aspect_ratio < 1:   
            return 1
        else: 
            return 2
    if are_numbers_close(uppRa, midRa, lowRa, threshold=0.1):
        if aspect_ratio < 3:   
            return 4
        else: 
            return 5
    if midRa > uppRa and midRa > lowRa:
        return 3
    if are_numbers_close(midRa, midRa, lowRa, threshold=0.1) and midRa > uppRa:
        return 8
    if uppRa > midRa and uppRa > lowRa:
        return 10
    for contour in contours:
        if is_smooth_contour(contour):
            return 9
    if aspect_ratio > 1.6:
        return 6
    return 7
            


def is_smooth_contour(contour, threshold=0.02):
    """
    判断轮廓是否是圆滑的曲线。
    
    参数:
    contour: 轮廓点集。
    threshold: 圆滑度阈值，默认为0.02。
    
    返回:
    如果轮廓是圆滑的曲线，返回True，否则返回False。
    """
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    
    # 计算圆形度
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    # 如果圆形度接近1，则认为是圆滑的曲线
    return circularity > (1 - threshold)

    
def are_numbers_close(a, b, c, threshold=0.1):
    """
    检查三个数之间的差值是否都不超过给定的阈值。

    参数:
    a, b, c: 要检查的三个数。
    threshold: 差值的阈值，默认为0.1。

    返回:
    如果三个数之间的差值都不超过阈值，返回True，否则返回False。
    """
    return abs(a - b) <= threshold and abs(b - c) <= threshold and abs(a - c) <= threshold


# 示例用法
image_path = 'centered_22A-T-36-10球形侧视图.JPG'  # 替换为你的图片路径


aspect_ratio, non_black_counts = calculate_image_metrics(image_path)
print(f"Aspect ratio (height/width) of the image: {aspect_ratio}")
print(f"Non-black pixel counts in each quarter section: {non_black_counts}")