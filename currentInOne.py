import cv2
import numpy as np
import os
import csv

green_lower = np.array([30, 40, 20])
green_upper = np.array([90, 255, 255])

white_lower1 = np.array([0, 0, 180])
white_upper1 = np.array([180, 45, 255])
white_lower2 = np.array([90, 0, 190])
white_upper2 = np.array([180, 130, 255])
white_lower3 = np.array([0, 0, 180])
white_upper3 = np.array([27, 130, 255])


def crop_center(image_path, output_filename):
    image = cv2.imread(image_path)
    cropped_image = image[0:3061, 1238:4343]
    cv2.imwrite(output_filename, cropped_image)
    return output_filename

def remove_white_rectangles(image_path, output_path):
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
        mask = np.ones_like(image) * 255
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
        #cv2.drawContours(image, contours_white, -1, (255, 0, 0), 1)
        #cv2.drawContours(image, [max_contour], -1, (222, 72, 138), 1)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [max_contour], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        extracted_image = cv2.bitwise_and(image, mask)
        cv2.imwrite(output_filename, extracted_image)
        return output_filename
    except Exception as e:
        print(f"Error finding and drawing contours: {e}")
        return None


def center_cabbage(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 找到非黑色像素（灰度值大于0的像素点）
    non_black_pixels = np.where(gray > 0)
    
    # 获取非黑色像素的最上、最下、最左和最右的位置
    top_y = np.min(non_black_pixels[0])    # 最上端非黑色像素的y坐标
    bottom_y = np.max(non_black_pixels[0]) # 最下端非黑色像素的y坐标
    left_x = np.min(non_black_pixels[1])   # 最左端非黑色像素的x坐标
    right_x = np.max(non_black_pixels[1])  # 最右端非黑色像素的x坐标
    
    # 裁剪图像，仅保留从最上到最下、最左到最右的非黑色区域
    cropped_image = image[top_y:bottom_y+1, left_x:right_x+1]
    
    # 保存结果图片
    cv2.imwrite(output_path, cropped_image)
    
    return cropped_image



def process_file(image_path, output_folder):
    try:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 创建文件夹
            cropped_folder_path = os.path.join(output_folder, 'cropped')
            if not os.path.exists(cropped_folder_path):
                os.makedirs(cropped_folder_path)
            
            extracted_folder_path = os.path.join(output_folder, 'extracted')
            if not os.path.exists(extracted_folder_path):
                os.makedirs(extracted_folder_path)
            
            center_folder_path = os.path.join(output_folder, 'center')
            if not os.path.exists(center_folder_path):
                os.makedirs(center_folder_path)
            
            base_filename = os.path.basename(image_path)
            
            # 裁剪图像
            cropped_image_path = os.path.join(cropped_folder_path, f"cropped_{base_filename}")
            crop_center(image_path, cropped_image_path)
            
            # 提取轮廓
            extracted_image_path = os.path.join(extracted_folder_path, f"extracted_{base_filename}")
            find_and_draw_contours(cropped_image_path, extracted_image_path)
            
            # 将提取后的图片居中
            centered_image_path = os.path.join(center_folder_path, f"centered_{base_filename}")
            center_cabbage(extracted_image_path, centered_image_path)
    except Exception as e:
        print(f"Error processing file: {e}")


def main(output_folder):
    try:
        output_folder_path = os.path.join(os.getcwd(), output_folder)
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        csv_path = os.path.join(output_folder_path, 'image_color_ratios.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Green Ratio', 'White Ratio'])
        for file in os.listdir():
            if file.lower().endswith(('.jpg', '.png')):
                process_file(file, output_folder_path)
    except Exception as e:
        print(f"Error in main function: {e}")






if __name__ == "__main__":
    main('output')