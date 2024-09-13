import cv2
from PIL import Image
import numpy as np

# def detect_leaf_handles(image_path, threshold=220, min_group_size=30, output_path='leaf_handle_lines_only.jpg'):
#     # Load the image
#     image = Image.open(image_path)
#     image_array = np.array(image)
#     height, width = image_array.shape[:2]

#     # Create a blank (black) image of the same size to draw the lines
#     lines_only_image = Image.new('RGB', (width, height), (0, 0, 0))

#     # Iterate over the rows of the image to find the white pixel groups
#     for row in range(height):
#         white_pixels = np.where(image_array[row] > threshold)[0]
        
#         # Group white pixels with small gaps or consecutive white pixels
#         if len(white_pixels) > min_group_size:
#             groups = np.split(white_pixels, np.where(np.diff(white_pixels) > 4)[0] + 1)
#             for group in groups:
#                 if len(group) >= min_group_size:
#                     # Draw the middle point of each group
#                     midpoint = group[len(group) // 2]
#                     lines_only_image.putpixel((midpoint, row), (255, 0, 0))

#     # Save the resulting image with only the marked lines
#     lines_only_image.save(output_path)
#     return output_path

def detect_leaf_handles(image_path, threshold=220, min_group_ratio=0.035, output_path='leaf_handle_lines_only.jpg'):
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    # Calculate the minimum group size based on the width of the image
    min_group_size = int(width * min_group_ratio)

    # Create a blank (black) image of the same size to draw the lines
    lines_only_image = Image.new('RGB', (width, height), (0, 0, 0))

    # Iterate over the rows of the image to find the white pixel groups
    for row in range(height):
        white_pixels = np.where(image_array[row] > threshold)[0]
        
        # Group white pixels with small gaps or consecutive white pixels
        if len(white_pixels) > min_group_size:
            groups = np.split(white_pixels, np.where(np.diff(white_pixels) > 4)[0] + 1)
            for group in groups:
                if len(group) >= min_group_size:
                    # Draw the middle point of each group
                    midpoint = group[len(group) // 2]
                    lines_only_image.putpixel((midpoint, row), (255, 0, 0))

    # Save the resulting image with only the marked lines
    lines_only_image.save(output_path)
    return output_path

# Example usage
# output_path = detect_leaf_handles('centered_22A-T-34-2球形侧视图.JPG')
# print(f"Output saved to: {output_path}")


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


def extract_white_veins_with_stem(image_path, threshold=220, output_path='white_veins_with_stem.jpg'):
    """
    Extracts white veins and draws the stem as a straight line from a cabbage image.

    Parameters:
    image_path (str): The path to the input cabbage image.
    threshold (int): The pixel intensity threshold to consider a pixel as white. Default is 220.
    output_path (str): The path to save the output image with extracted white veins and stem. Default is 'white_veins_with_stem.jpg'.

    Returns:
    str: The path to the saved output image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the contours
    veins_image = np.zeros_like(image)

    # Draw the contours on the blank image
    cv2.drawContours(veins_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convert the veins image to grayscale
    veins_gray = cv2.cvtColor(veins_image, cv2.COLOR_BGR2GRAY)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(veins_gray, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw the detected lines on the veins image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(veins_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save the resulting image with extracted white veins and stem
    cv2.imwrite(output_path, veins_image)
    return output_path

# Example usage
extract_white_veins_with_stem('centered_22A-T-34-2球形侧视图.JPG')




# 读取图像
image = cv2.imread(image_path)
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