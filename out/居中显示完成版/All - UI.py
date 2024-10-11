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
















# Function to select an image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # Process the image and display results
        result = process_image(file_path)
        if result:
            result_text.set(f"Filename: {result[0]}\nGreen Ratio: {result[1]:.2%}\nWhite Ratio: {result[2]:.2%}\nBall Shape: {result[3]}\nHug Type: {result[4]}")
        else:
            result_text.set("Error processing the image.")

# Function to apply a filter to the image
def apply_filter(filter_type):
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        if filter_type == "BLUR":
            pil_img = pil_img.filter(ImageFilter.BLUR)
        elif filter_type == "CONTOUR":
            pil_img = pil_img.filter(ImageFilter.CONTOUR)
        elif filter_type == "DETAIL":
            pil_img = pil_img.filter(ImageFilter.DETAIL)
        elif filter_type == "EDGE_ENHANCE":
            pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_type == "SHARPEN":
            pil_img = pil_img.filter(ImageFilter.SHARPEN)
        
        img = ImageTk.PhotoImage(pil_img)
        panel.config(image=img)
        panel.image = img

# Function to rotate the image
def rotateaaaa_image(angle):
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        pil_img = pil_img.rotate(angle)
        img = ImageTk.PhotoImage(pil_img)
        panel.config(image=img)
        panel.image = img

# Function to scale the image
def scaleaa_image(scale_factor):
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        width, height = pil_img.size
        pil_img = pil_img.resize((int(width * scale_factor), int(height * scale_factor)))
        img = ImageTk.PhotoImage(pil_img)
        panel.config(image=img)
        panel.image = img

# Function to show histogram
def show_histogram():
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        img_array = np.array(pil_img)
        plt.figure()
        plt.hist(img_array.ravel(), bins=256, color='orange', )
        plt.hist(img_array[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
        plt.hist(img_array[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
        plt.hist(img_array[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
        plt.show()

# Function to save the processed image
def saveaa_image():
    if panel.image:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
        if file_path:
            img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
            pil_img = Image.frombytes("RGB", img.size, img.tobytes())
            pil_img.save(file_path)
            messagebox.showinfo("Image Saved", f"Image saved as {file_path}")

# Function to adjust brightness
def adjustaa_brightness(factor):
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(factor)
        img = ImageTk.PhotoImage(pil_img)
        panel.config(image=img)
        panel.image = img

# Function to adjust contrast
def adjustaa_contrast(factor):
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(factor)
        img = ImageTk.PhotoImage(pil_img)
        panel.config(image=img)
        panel.image = img

# Function to adjust brightness
def adjust_brightness(factor):
    if panel.image:
        img = panel.image._PhotoImage__photo.zoom(1)  # Get the original image
        pil_img = Image.frombytes("RGB", img.size, img.tobytes())
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(factor)
        img = ImageTk.PhotoImage(pil_img)
        panel.config(image=img)
        panel.image = img

# Function to crop image
def crop_image(image_path, crop_box, output_filename):
    """
    Crop the image to the specified box and save the result.
    """
    image = Image.open(image_path)
    cropped_image = image.crop(crop_box)
    cropped_image.save(output_filename)
    print(f"Image saved as {output_filename}")

# Function to convert image to grayscale
def convert_to_grayscale(image_path, output_filename):
    """
    Convert the image to grayscale and save the result.
    """
    image = Image.open(image_path)
    grayscale_image = image.convert("L")
    grayscale_image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
def flip_image(image_path, direction, output_filename):
    """
    Flip the image in the specified direction ('horizontal' or 'vertical') and save the result.
    """
    image = Image.open(image_path)
    if direction == 'horizontal':
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    flipped_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def blur_image(image_path, radius, output_filename):
    """
    Apply a blur filter to the image with the specified radius and save the result.
    """
    image = Image.open(image_path)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    blurred_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def edge_detect_image(image_path, output_filename):
    """
    Apply an edge detection filter to the image and save the result.
    """
    image = Image.open(image_path)
    edge_detected_image = image.filter(ImageFilter.FIND_EDGES)
    edge_detected_image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
def scale_image(image_path, scale_factor, output_filename):
    """
    Scale the image by the specified factor and save the result.
    """
    image = Image.open(image_path)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    scaled_image = image.resize(new_size, Image.ANTIALIAS)
    scaled_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def sharpena_image(image_path, output_filename):
    """
    Apply a sharpen filter to the image and save the result.
    """
    image = Image.open(image_path)
    sharpened_image = image.filter(ImageFilter.SHARPEN)
    sharpened_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def adjust_contrast(image_path, factor, output_filename):
    """
    Adjust the contrast of the image by the specified factor and save the result.
    """
    image = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(image)
    contrast_image = enhancer.enhance(factor)
    contrast_image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
def rotateaa_image(image_path, angle, output_filename):
    """
    Rotate the image by a specified angle and save the result.
    """
    image = Image.open(image_path)
    rotated_image = image.rotate(angle)
    rotated_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def mirroraa_image(image_path, output_filename):
    """
    Create a mirror image (flip horizontally) and save the result.
    """
    image = Image.open(image_path)
    mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def invertaa_colors(image_path, output_filename):
    """
    Invert the colors of the image and save the result.
    """
    image = Image.open(image_path)
    inverted_image = ImageOps.invert(image.convert("RGB"))
    inverted_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def adjust_saturation(image_path, factor, output_filename):
    """
    Adjust the saturation of the image by the specified factor and save the result.
    """
    image = Image.open(image_path)
    enhancer = ImageEnhance.Color(image)
    saturated_image = enhancer.enhance(factor)
    saturated_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def smooth_image(image_path, output_filename):
    """
    Apply a smoothing filter to the image and save the result.
    """
    image = Image.open(image_path)
    smoothed_image = image.filter(ImageFilter.SMOOTH)
    smoothed_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def contour_image(image_path, output_filename):
    """
    Apply a contour filter to the image and save the result.
    """
    image = Image.open(image_path)
    contoured_image = image.filter(ImageFilter.CONTOUR)
    contoured_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def add_noise(image_path, output_filename, noise_level=0.1):
    """
    Add random noise to the image and save the result.
    """
    image = Image.open(image_path)
    np_image = np.array(image)
    noise = np.random.normal(0, noise_level, np_image.shape)
    noisy_image = np.clip(np_image + noise * 255, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)
    noisy_image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
def rotatea_image(image_path, angle, output_filename):
    """
    Rotate the image by a specified angle and save the result.
    """
    image = Image.open(image_path)
    rotated_image = image.rotate(angle)
    rotated_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def mirror_image(image_path, output_filename):
    """
    Create a mirror image (flip horizontally) and save the result.
    """
    image = Image.open(image_path)
    mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    mirrored_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def invert_colors(image_path, output_filename):
    """
    Invert the colors of the image and save the result.
    """
    image = Image.open(image_path)
    inverted_image = ImageOps.invert(image.convert("RGB"))
    inverted_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def add_watermark(image_path, watermark_path, position, output_filename):
    """
    Add a watermark to the image at the specified position and save the result.
    """
    image = Image.open(image_path)
    watermark = Image.open(watermark_path).convert("RGBA")
    image.paste(watermark, position, watermark)
    image.save(output_filename)
    print(f"Image saved as {output_filename}")

def rotate_image(image_path, angle, output_filename):
    """
    Rotate the image by a specified angle and save the result.
    """
    image = Image.open(image_path)
    rotated_image = image.rotate(angle)
    rotated_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def resize_image(image_path, size, output_filename):
    """
    Resize the image to the specified size and save the result.
    """
    image = Image.open(image_path)
    resized_image = image.resize(size)
    resized_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def sharpen_image(image_path, output_filename):
    """
    Sharpen the image and save the result.
    """
    image = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(2.0)  # Increase sharpness by a factor of 2
    sharpened_image.save(output_filename)
    print(f"Image saved as {output_filename}")

def save_image(image, output_path):
    """
    Save the image to the specified path.
    """
    image.save(output_path)
    print(f"Image saved as {output_path}")

def open_image():
    global img, panel
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

def rotate_image_ui():
    global img
    if img:
        rotated_img = img.rotate(45)
        img_tk = ImageTk.PhotoImage(rotated_img)
        panel.config(image=img_tk)
        panel.image = img_tk

def resize_image_ui():
    global img
    if img:
        resized_img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(resized_img)
        panel.config(image=img_tk)
        panel.image = img_tk

def sharpen_image_ui():
    global img
    if img:
        enhancer = ImageEnhance.Sharpness(img)
        sharpened_img = enhancer.enhance(2.0)
        img_tk = ImageTk.PhotoImage(sharpened_img)
        panel.config(image=img_tk)
        panel.image = img_tk



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
root.title("Cabbage Image Processor")

# 创建显示图像的面板
panel = tk.Label(root)
panel.pack()

# 加载并显示默认图像
default_image_path = 'default_cabbage.jpg'
if os.path.exists(default_image_path):
    default_img = Image.open(default_image_path)
    default_img.thumbnail((400, 400))
    default_img = ImageTk.PhotoImage(default_img)
    panel.config(image=default_img)
    panel.image = default_img

# 创建按钮以选择图像
btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack()

# 创建应用滤镜的按钮
btn_blur = tk.Button(root, text="Blur", command=lambda: apply_filter("BLUR"))
btn_blur.pack()
btn_contour = tk.Button(root, text="Contour", command=lambda: apply_filter("CONTOUR"))
btn_contour.pack()
btn_detail = tk.Button(root, text="Detail", command=lambda: apply_filter("DETAIL"))
btn_detail.pack()
btn_edge_enhance = tk.Button(root, text="Edge Enhance", command=lambda: apply_filter("EDGE_ENHANCE"))
btn_edge_enhance.pack()
btn_sharpen = tk.Button(root, text="Sharpen", command=lambda: apply_filter("SHARPEN"))
btn_sharpen.pack()

# 创建新功能的按钮
btn_rotate_left = tk.Button(root, text="Rotate Left", command=lambda: rotateaaaa_image(90))
btn_rotate_left.pack()
btn_rotate_right = tk.Button(root, text="Rotate Right", command=lambda: rotateaaaa_image(-90))
btn_rotate_right.pack()
btn_scale_up = tk.Button(root, text="Scale Up", command=lambda: scaleaa_image(1.2))
btn_scale_up.pack()
btn_scale_down = tk.Button(root, text="Scale Down", command=lambda: scaleaa_image(0.8))
btn_scale_down.pack()
btn_histogram = tk.Button(root, text="Show Histogram", command=show_histogram)
btn_histogram.pack()
btn_save = tk.Button(root, text="Save Image", command=saveaa_image)
btn_save.pack()
btn_brightness_up = tk.Button(root, text="Brightness Up", command=lambda: adjust_brightness(1.2))
btn_brightness_up.pack()
btn_brightness_down = tk.Button(root, text="Brightness Down", command=lambda: adjust_brightness(0.8))
btn_brightness_down.pack()
btn_contrast_up = tk.Button(root, text="Contrast Up", command=lambda: adjustaa_contrast(1.2))
btn_contrast_up.pack()
btn_contrast_down = tk.Button(root, text="Contrast Down", command=lambda: adjustaa_contrast(0.8))
btn_contrast_down.pack()

# 创建处理当前文档的按钮
btn_deal_current_docu = tk.Button(root, text="Deal Current Docu", command=deal_current_docu)
btn_deal_current_docu.pack()

# 创建显示结果的标签
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify=tk.LEFT)
result_label.pack()

# 运行应用程序
root.mainloop()