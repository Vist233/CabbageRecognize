import cv2
import numpy as np



# 区分14 23 ：23 比例高，为60+%
def calculate_non_black_ratio(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    for y in range(height):
        row = gray[y, :]
        
        # Count non-black pixels in the row
        non_black_pixels = np.count_nonzero(row)
        total_pixels = width
        
        # Calculate the ratio of non-black pixels
        non_black_ratio = non_black_pixels / total_pixels
        
        if non_black_ratio >= 0.6:
            # Find the middle element position
            middle_position = width // 2
            
            # Find the leftmost and rightmost non-black pixels
            leftmost = np.argmax(row > 0)
            rightmost = width - 1 - np.argmax(np.flip(row) > 0)
            
            # Define the rectangle from the leftmost to the rightmost pixel
            topmost = y
            bottommost = min(leftmost, rightmost)
            
            # Extract the rectangle region
            rectangle = gray[topmost:bottommost+1, leftmost:rightmost+1]
            
            # Calculate the non-black ratio in the rectangle
            non_black_pixels_rect = np.count_nonzero(rectangle)
            total_pixels_rect = rectangle.size
            non_black_ratio_rect = non_black_pixels_rect / total_pixels_rect
            
            print(f"Row {y}: Non-black ratio in rectangle = {non_black_ratio_rect:.2f}")

            break



# 区分1 4：4的比例大 大约为4+
def calculate_perimeter(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Crop the top third of the image
    cropped_gray = gray[:height // 3, :]
    
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(cropped_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the perimeter of the largest contour
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Draw the contour on the original image for visualization
        output_image = cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)
        
        # Save the output image
        cv2.imwrite('output_image.jpg', output_image)
        
        return perimeter / width
    else:
        return 0

# Load the image
image = cv2.imread('centered_22A-T-34-2球形侧视图.JPG')

# Calculate the non-black ratio
calculate_non_black_ratio(image)
print(calculate_perimeter(image))