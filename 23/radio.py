import cv2

def calculate_aspect_ratio(image_path):
    # 读取输入图片
    image = cv2.imread(image_path)
    
    # 获取图片的尺寸
    height, width = image.shape[:2]
    
    # 计算宽高比
    aspect_ratio = height / width
    
    return aspect_ratio

# 示例用法
image_path = 'largest_white_area.jpg'  # 替换为你的图片路径
aspect_ratio = calculate_aspect_ratio(image_path)
print(f"{aspect_ratio}")