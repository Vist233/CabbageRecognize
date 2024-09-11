import cv2
from PIL import Image
import numpy as np

# def extract_white_veins(image_path, threshold=220, output_path='white_veins.jpg'):
#     """直接叶脉
#     Extracts white veins from a cabbage image.

#     Parameters:
#     image_path (str): The path to the input cabbage image.
#     threshold (int): The pixel intensity threshold to consider a pixel as white. Default is 220.
#     output_path (str): The path to save the output image with extracted white veins. Default is 'white_veins.jpg'.

#     Returns:
#     str: The path to the saved output image.
#     """
#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or unable to read")

#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply threshold to get binary image
#     _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

#     # Find contours
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create a blank image to draw the contours
#     veins_image = np.zeros_like(image)

#     # Draw the contours on the blank image
#     cv2.drawContours(veins_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

#     # Save the resulting image with extracted white veins
#     cv2.imwrite(output_path, veins_image)
#     return output_path
# # Example usage
# extract_white_veins('centered_22A-T-34-2球形侧视图.JPG')

if image is None:
    raise ValueError("Image not found or unable to read")

# 将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值处理得到二值图像
_, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像用于绘制轮廓
veins_image = np.zeros_like(image)

# 在空白图像上绘制轮廓
cv2.drawContours(veins_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# 将叶脉图像转换为灰度
veins_gray = cv2.cvtColor(veins_image, cv2.COLOR_BGR2GRAY)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(veins_gray, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# 在叶脉图像上绘制检测到的直线
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(veins_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 保存提取出白色叶脉和叶柄的结果图像
cv2.imwrite(output_path, veins_image)
return output_path

# 读取图像
image = cv2.imread(image_path)


