import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import cv2
import numpy as np
import os
import csv
import logging
import matplotlib.pyplot as plt

green_lower = np.array([30, 40, 20])
green_upper = np.array([90, 255, 255])

white_lower1 = np.array([0, 0, 180])
white_upper1 = np.array([180, 22, 255])
white_lower2 = np.array([90, 0, 190])
white_upper2 = np.array([180, 130, 255])
white_lower3 = np.array([0, 0, 180])
white_upper3 = np.array([27, 130, 255])

def are_numbers_close(a, b, c, threshold=0.1):
    return abs(a - b) <= threshold and abs(b - c) <= threshold and abs(a - c) <= threshold

def is_smooth_contour(contour, threshold=0.02):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity > (1 - threshold)

def cropBlackCabbage(image_path, output_filename):
    try:
        image = cv2.imread(image_path)
        cropped_image = image[0:3061, 1238:4343]
        cv2.imwrite(output_filename, cropped_image)
        return output_filename
    except Exception as e:
        print(f"Error cropping black cabbage: {e}")
        return None

def remove_white_Background(image_path, output_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for contour in contours:
            epsilon = 0.10 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                rectangles.append(approx)
        mask = np.ones_like(image) * 255 # type: ignore
        cv2.drawContours(mask, rectangles, -1, (0, 0, 0), thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, mask)
        cv2.imwrite(output_path, result)
        print(f"Processed image saved as {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

def find_and_draw_contours(image_path, output_filename):
    try:
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        white_mask1 = cv2.inRange(hsv, white_lower1, white_upper1)
        white_mask2 = cv2.inRange(hsv, white_lower2, white_upper2)
        white_mask3 = cv2.inRange(hsv, white_lower3, white_upper3)
        white_mask = cv2.bitwise_or(white_mask1, white_mask2)
        white_mask = cv2.bitwise_or(white_mask, white_mask3)

        mask_combined = cv2.bitwise_or(mask_green, white_mask)
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found")
            return
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        # cv2.drawContours(mask, [max_contour], -1, color=(0, 0, 0), thickness=cv2.FILLED)
        extracted_image = cv2.bitwise_and(image, mask)
        cv2.imwrite(output_filename, extracted_image)
        return output_filename
    except Exception as e:
        print(f"Error finding and drawing contours: {e}")
        return None

def calculate_color_proportion(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像文件。请检查路径。")
            return [0, 0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_mask1 = cv2.inRange(hsv, white_lower1, white_upper1)
        white_mask2 = cv2.inRange(hsv, white_lower2, white_upper2)
        white_mask3 = cv2.inRange(hsv, white_lower3, white_upper3)
        white_mask = cv2.bitwise_or(white_mask1, white_mask2)
        white_mask = cv2.bitwise_or(white_mask, white_mask3)

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = cv2.countNonZero(green_mask)
        white_pixels = cv2.countNonZero(white_mask)
        
        non_black_mask = cv2.inRange(hsv, np.array([0, 0, 1]), np.array([180, 255, 255]))
        non_black_pixels = cv2.countNonZero(non_black_mask)

        if non_black_pixels > 0:
            white_ratio = white_pixels / non_black_pixels
            green_ratio = 1 - white_ratio
        else:
            print("没有找到符合条件的像素。")
            green_ratio = 0
            white_ratio = 0
        
        print(f"菜叶比例: {green_ratio:.2%}")
        print(f"菜帮比例: {white_ratio:.2%}")
        return [green_ratio, white_ratio]
    except Exception as e:
        print(f"Error calculating color proportion: {e}")
        return [0, 0]

def getCabbageInCenter(image_path, output_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_black_pixels = np.where(gray > 0)
        top_y = np.min(non_black_pixels[0])
        bottom_y = np.max(non_black_pixels[0])
        left_x = np.min(non_black_pixels[1])
        right_x = np.max(non_black_pixels[1])
        cropped_image = image[top_y:bottom_y+1, left_x:right_x+1]
        cv2.imwrite(output_path, cropped_image)
        return cropped_image
    except Exception as e:
        print(f"Error getting cabbage in center: {e}")
        return None

def hug23ImportantAspect(image_path, output_path):
    try:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        half_image = image[:, :width // 2]
        hsv_image = cv2.cvtColor(half_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv_image, white_lower1, white_upper1)
        mask2 = cv2.inRange(hsv_image, white_lower2, white_upper2)
        mask3 = cv2.inRange(hsv_image, white_lower3, white_upper3)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            largest_white_area = np.zeros_like(half_image)
            cv2.drawContours(largest_white_area, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            cv2.imwrite(output_path, largest_white_area)
        else:
            print("No white areas found.")
    except Exception as e:
        print(f"Error in hug23ImportantAspect: {e}")

def calculate_pic_ratio(image_path):
    try:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        aspect_ratio = height / width
        return aspect_ratio
    except Exception as e:
        print(f"Error calculating picture ratio: {e}")
        return None

def BallShapeOUT(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        aspect_ratio = height / width
        non_black_threshold = 10
        quarter_height = height // 4
        non_black_counts = []
        for i in range(1, 4):
            section = image[(i-1)*quarter_height:i*quarter_height, :]
            non_black_count = np.sum(np.any(section > non_black_threshold, axis=2))
            non_black_counts.append(non_black_count)
        
        sum_counts = non_black_counts[0] + non_black_counts[1] + non_black_counts[2]
        uppRa = non_black_counts[0] / sum_counts 
        midRa = non_black_counts[1] / sum_counts
        lowRa = non_black_counts[2] / sum_counts

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

        if are_numbers_close(midRa, lowRa, uppRa, threshold=0.1) and midRa > uppRa:
            return 8

        if uppRa > midRa and uppRa > lowRa:
            return 10

        for contour in contours:
            if is_smooth_contour(contour):
                return 9

        if aspect_ratio > 1.6:
            return 6

        return 7
    except Exception as e:
        print(f"Error in BallShapeOUT: {e}")
        return None

def calculate_perimeter_Curve_radio(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        cropped_img = img.crop((0, 0, width, height // 3))
        gray_img = cropped_img.convert('L')
        img_array = np.array(gray_img)
        threshold = 50
        binary_img = img_array > threshold
        contours = measure.find_contours(binary_img, 0.8)
        total_perimeter = sum([measure.perimeter(c) for c in contours])
        perimeter_to_width_ratio = total_perimeter / width
        return perimeter_to_width_ratio
    except Exception as e:
        print(f"Error calculating perimeter curve ratio: {e}")
        return None

# Setup logging
logging.basicConfig(filename='cabbage_processor.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the directories
input_folder = './'
output_folder = './output/Cabbage'
center_output_folder = './output/center'
hug_output_folder = './output/hug'

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(center_output_folder, exist_ok=True)
os.makedirs(hug_output_folder, exist_ok=True)

# Function to process the image and get results
def process_image(image_path):
    try:
        logging.info(f"Processing image: {image_path}")
        contour_output_path = os.path.join(output_folder, os.path.basename(image_path))
        center_image_path = os.path.join(center_output_folder, os.path.basename(image_path))
        hug_image_path = os.path.join(hug_output_folder, os.path.basename(image_path))
        
        find_and_draw_contours(image_path, contour_output_path)
        color_proportion = calculate_color_proportion(contour_output_path)
        row = [os.path.basename(image_path), color_proportion[0], color_proportion[1]]

        getCabbageInCenter(contour_output_path, center_image_path)
        ball_shape = BallShapeOUT(center_image_path)
        row.append(ball_shape)

        curve_ratio = calculate_perimeter_Curve_radio(center_image_path)
        if curve_ratio is None:
            row.append("Unknown")
        elif curve_ratio < 2:
            row.append("叠抱")
        elif curve_ratio > 4:
            row.append("翻心")
        else:
            hug23ImportantAspect(center_image_path, hug_image_path)
            pic_ratio = calculate_pic_ratio(hug_image_path)
            if pic_ratio is None:
                row.append("Unknown")
            elif pic_ratio < 3:
                row.append("合抱")
            else:
                row.append("拧抱")

        return row
    except Exception as e:
        logging.error(f"Error processing file {image_path}: {e}")
        return None




# 定义处理当前文档的函数
def deal_current_docu():
    # 定义目录
    input_folder = './'
    output_folder = './output/Cabbage'
    center_output_folder = './output/center'
    hug_output_folder = './output/hug'

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(center_output_folder, exist_ok=True)
    os.makedirs(hug_output_folder, exist_ok=True)

    # 列出当前目录中的所有PNG文件
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # 初始化CSV输出
    csv_output_path = './output/outcome.csv'
    with open(csv_output_path, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Green Radio', 'White Radio', 'Ball Shape', 'Hug Type'])

        for png_file in png_files:
            try:
                input_image_path = os.path.join(input_folder, png_file)
                contour_output_path = os.path.join(output_folder, png_file)
                center_image_path = os.path.join(center_output_folder, png_file)
                hug_image_path = os.path.join(hug_output_folder, png_file)
                
                find_and_draw_contours(input_image_path, contour_output_path)
                color_proportion = calculate_color_proportion(contour_output_path)
                row = [png_file, color_proportion[0], color_proportion[1]]

                getCabbageInCenter(contour_output_path, center_image_path)
                ball_shape = BallShapeOUT(center_image_path)
                row.append(ball_shape)

                curve_ratio = calculate_perimeter_Curve_radio(center_image_path)
                if curve_ratio is None:
                    row.append("Unknown")
                elif curve_ratio < 2:
                    row.append("叠抱")
                elif curve_ratio > 4:
                    row.append("翻心")
                else:
                    hug23ImportantAspect(center_image_path, hug_image_path)
                    pic_ratio = calculate_pic_ratio(hug_image_path)
                    if pic_ratio is None:
                        row.append("Unknown")
                    elif pic_ratio < 3:
                        row.append("合抱")
                    else:
                        row.append("拧抱")

                csvwriter.writerow(row)
            except Exception as e:
                print(f"Error processing file {png_file}: {e}")



# 创建主窗口
root = tk.Tk()
root.title("UI Example")

# 创建左侧和右侧的Frame
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# 显示默认图像
default_image_path = 'default_cabbage.jpg'
if os.path.exists(default_image_path):
    default_img = Image.open(default_image_path)
    default_img.thumbnail((400, 400))
    default_img = ImageTk.PhotoImage(default_img)
    panel = tk.Label(left_frame, image=default_img)
    panel.image = default_img
    panel.pack()


# 在左侧Frame中创建按钮
btn_histogram = tk.Button(left_frame, text="Histogram", command=lambda: print("Histogram"))
btn_histogram.pack()
btn_save = tk.Button(left_frame, text="Save Image", command=lambda: print("Save Image"))
btn_save.pack()
btn_brightness_up = tk.Button(left_frame, text="Brightness Up", command=lambda: print("Brightness Up"))
btn_brightness_up.pack()
btn_brightness_down = tk.Button(left_frame, text="Brightness Down", command=lambda: print("Brightness Down"))
btn_brightness_down.pack()
btn_contrast_up = tk.Button(left_frame, text="Contrast Up", command=lambda: print("Contrast Up"))
btn_contrast_up.pack()
btn_contrast_down = tk.Button(left_frame, text="Contrast Down", command=lambda: print("Contrast Down"))
btn_contrast_down.pack()

# 创建处理当前文档的按钮
btn_deal_current_docu = tk.Button(left_frame, text="Deal Current Docu", command=deal_current_docu)
btn_deal_current_docu.pack()

# 在右侧Frame中创建显示结果的标签
result_text = tk.StringVar()
result_label = tk.Label(right_frame, textvariable=result_text, justify=tk.LEFT)
result_label.pack()


# 运行应用程序
root.mainloop()