from PIL import Image
import numpy as np
from skimage import measure

def calculate_perimeter_Curve_radio(image_path):
    # 加载输入图片
    img = Image.open(image_path)

    # 获取图片的尺寸
    width, height = img.size

    # 裁剪上三分之一的图像
    cropped_img = img.crop((0, 0, width, height // 3))

    # 转换为灰度图像
    gray_img = cropped_img.convert('L')

    # 转换为numpy数组
    img_array = np.array(gray_img)

    # 应用阈值以区分背景（黑色）和对象（有色）
    threshold = 50  # 定义阈值
    binary_img = img_array > threshold  # 二值图像：对象为True，背景为False

    # 在二值图像中找到轮廓
    contours = measure.find_contours(binary_img, 0.8)

    # 计算所有轮廓的总周长
    total_perimeter = sum([measure.perimeter(c) for c in contours])

    # 计算周长与原始图像宽度的比值
    perimeter_to_width_ratio = total_perimeter / width

    return perimeter_to_width_ratio

# 示例用法
image_path = 'centered_22A-T-39-1球形侧视图.JPG'  # 替换为你的图片路径
perimeter_ratio = calculate_perimeter_ratio(image_path)
print(f"Perimeter to width ratio: {perimeter_ratio}")