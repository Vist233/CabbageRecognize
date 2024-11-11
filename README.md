# Cabbage Image Processing Pipeline

## Introduction

This project is a Python-based image processing tool designed to analyze and classify cabbage images. Utilizing libraries such as OpenCV, NumPy, PIL, and scikit-image, the program crops images, removes white backgrounds, extracts contours, calculates color proportions, identifies ball shapes, and determines hugging types. Results are exported to a CSV file for easy data analysis.

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pillow
- scikit-image

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/cabbage-image-processing.git
    cd cabbage-image-processing
    ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *`requirements.txt`:*

    ```plaintext
    opencv-python
    numpy
    pillow
    scikit-image
    ```

## Usage

1. **Place PNG Images:**

    Add the PNG images you want to process to the project's root directory.

2. **Run the Script:**

    ```bash
    python your_script.py
    ```

    The script will process all PNG files in the current directory and save the results in the `./output` folder.

## Features

- **Crop Black Cabbage Parts:** `cropBlackCabbage` function crops unwanted black areas from images.
- **Remove White Background:** `remove_white_Background` function removes white backgrounds using thresholding and contour detection.
- **Contour Detection and Drawing:** `find_and_draw_contours` function detects and draws the main contours in images.
- **Color Proportion Calculation:** `calculate_color_proportion` function calculates the ratio of green and white areas in the image.
- **Center Cabbage Extraction:** `getCabbageInCenter` function extracts the central part of the cabbage in the image.
- **Ball Shape Identification:** `BallShapeOUT` function identifies the ball shape of the cabbage based on shape features.
- **Perimeter Curve Ratio Calculation:** `calculate_perimeter_Curve_radio` function calculates the ratio of perimeter to width.
- **Hugging Type Determination:** Determines the type of cabbage hugging (e.g., stacked, twisted) based on multiple metrics.

## Output

- **Images:**
    - `./output/Cabbage/`: Processed contour images.
    - `./output/center/`: Central cabbage images.
    - `./output/hug/`: Hugging type related images.

- **CSV File:**
    - `./output/outcome.csv`: Contains filename, green ratio, white ratio, ball shape, and hugging type.

    *CSV Structure Example:*

    | Filename    | Green Ratio | White Ratio | Ball Shape | Hug Type |
    |-------------|-------------|-------------|------------|----------|
    | image1.png  | 75.00%      | 25.00%      | 1          | Stacked  |
    | image2.png  | 60.00%      | 40.00%      | 2          | Twisted  |
    | ...         | ...         | ...         | ...        | ...      |

## Example

For an image named `cabbage1.png`, running the script will generate:

- `./output/Cabbage/cabbage1.png`: Processed contour image.
- `./output/center/cabbage1.png`: Central cabbage image.
- `./output/hug/cabbage1.png`: Hugging type image.
- A corresponding entry in `./output/outcome.csv`.

## Contributing

Feel free to submit issues and pull requests to contribute to the project and improve its functionality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.