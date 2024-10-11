import cv2
import numpy as np
import os
import csv

green_lower = np.array([30, 40, 20])
green_upper = np.array([90, 255, 255])

white_lower1 = np.array([0, 0, 180])
white_upper1 = np.array([180, 22, 255])
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

import cv2
import numpy as np

def calculate_color_proportion(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像文件。请检查路径。")
            return
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
        cv2.drawContours(image, contours_white, -1, (255, 0, 0), 1)
        #cv2.drawContours(image, [max_contour], -1, (222, 72, 138), 1)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [max_contour], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        extracted_image = cv2.bitwise_and(image, mask)
        cv2.imwrite(output_filename, extracted_image)
        return output_filename
    except Exception as e:
        print(f"Error finding and drawing contours: {e}")
        return None

def process_file(image_path, output_folder):
    try:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create a folder for cropped images
            cropped_folder_path = os.path.join(output_folder, 'cropped')
            if not os.path.exists(cropped_folder_path):
                os.makedirs(cropped_folder_path)
            # Create a folder for extracted images
            extracted_folder_path = os.path.join(output_folder, 'extracted')
            if not os.path.exists(extracted_folder_path):
                os.makedirs(extracted_folder_path)
            
            base_filename = os.path.basename(image_path)
            cropped_image_path = os.path.join(cropped_folder_path, f"cropped_{base_filename}")
            # Crop the center of the image first
            crop_center(image_path, cropped_image_path)
            
            extracted_image_path = os.path.join(extracted_folder_path, f"extracted_{base_filename}")
            # Then find and draw contours on the cropped image
            find_and_draw_contours(cropped_image_path, extracted_image_path)
            
            # Calculate color proportions on the extracted image
            ratios = calculate_color_proportion(extracted_image_path)
            with open(os.path.join(output_folder, 'image_color_ratios.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_path] + ratios)
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