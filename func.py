def process_image(image_path, output_path):
    # 读取输入图片
    image = cv2.imread(image_path)

    # 获取图片的尺寸
    height, width = image.shape[:2]

    # 截取图片的前一半
    half_image = image[:, :width // 2]

    # 定义白色的HSV范围
    white_lower1 = np.array([0, 0, 180])
    white_upper1 = np.array([180, 45, 255])
    white_lower2 = np.array([90, 0, 190])
    white_upper2 = np.array([180, 130, 255])
    white_lower3 = np.array([0, 0, 180])
    white_upper3 = np.array([27, 130, 255])

    # 转换图片到HSV颜色空间
    hsv_image = cv2.cvtColor(half_image, cv2.COLOR_BGR2HSV)

    # 创建掩码
    mask1 = cv2.inRange(hsv_image, white_lower1, white_upper1)
    mask2 = cv2.inRange(hsv_image, white_lower2, white_upper2)
    mask3 = cv2.inRange(hsv_image, white_lower3, white_upper3)

    # 合并掩码
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    # 找到所有的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        # 创建一个空白的掩码
        largest_white_area = np.zeros_like(half_image)

        # 绘制最大的轮廓
        cv2.drawContours(largest_white_area, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # 保存结果到指定路径
        cv2.imwrite(output_path, largest_white_area)
    else:
        print("No white areas found.")




def calculate_aspect_ratio(image_path):
    # 读取输入图片
    image = cv2.imread(image_path)
    
    # 获取图片的尺寸
    height, width = image.shape[:2]
    
    # 计算宽高比
    aspect_ratio = height / width
    
    return aspect_ratio