# OpenCV Image Processing Application

A PyQt5-based application for demonstrating various image processing techniques using OpenCV. This project implements comprehensive image processing operations including color manipulation, image smoothing, edge detection, and geometric transformations.

## Features

### 1. Image Processing 
1.1. Color Separation 
1.2. Color Transformation 
1.3. Color Detection 
1.4. Blending 

### 2. Image Smoothing 
2.1. Gaussian Blur 
2.2. Bilateral Filter 
2.3. Median Filter 

### 3. Edge Detection
3.1. Gaussian Blur
3.2. Sobel X (Vertical Edge Detection)
3.3. Sobel Y (Horizontal Edge Detection)
3.4. Magnitude

### 4. Transforms
4.1. Resize
4.2. Translation
4.3. Rotation & Scaling
4.4. Shearing

## Requirements

### Python Dependencies
- Python 3.7
- opencv-contrib-python (3.4.2.17)
- Matplotlib 3.1.1
- PyQt5 (5.15.1)
- NumPy

### Required Images
- OpenCV.png (for color operations)
- Dog_Strong.jpg and Dog_Weak.jpg (for blending)
- beauty.png (for smoothing operations)
- building.jpg (for edge detection)
- Microsoft.png (for transforms)

## Installation and Setup

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Operation

1. Run the application:
```bash
python main.py
```

2. Using the Interface:
   - Use "Load Image 1" for single image operations
   - Use both "Load Image 1" and "Load Image 2" for blending
   - Click operation buttons to perform specific tasks
   - Use trackbars in popup windows to adjust parameters
   - Press 'Esc' to close popup windows

### Feature-Specific Operations

#### Edge Detection

1. Load building.jpg using "Load Image 1" button
2. Select operations in sequence:
   - "3.1 Gaussian Blur" for initial smoothing
   - "3.2 Sobel X" for vertical edges
   - "3.3 Sobel Y" for horizontal edges
   - "3.4 Magnitude" for combined edge strength

#### Transform Operations

1. Load Microsoft.png using "Load Image 1" button
2. Apply transformations in sequence:
   - "4.1 Resize" (430x430 → 215x215)
   - "4.2 Translation" (+215 pixels X/Y)
   - "4.3 Rotation, Scaling" (45°, 0.5x)
   - "4.4 Shearing" (affine transform)

## Technical Details

### Project Structure
```
project/
│
├── main.py           # Main application entry point
├── controller.py     # Main logic and image processing functions
├── UI.py            # PyQt5 UI definition
│
├── images/          # Sample images directory
│   ├── OpenCV.png
│   ├── Dog_Strong.jpg
│   ├── Dog_Weak.jpg
│   ├── beauty.png
│   ├── building.jpg
│   └── Microsoft.png
│
└── requirements.txt  # Project dependencies
```

### Implementation Details

#### Edge Detection Kernels

- Gaussian Kernel (3x3):
  ```
  [[0.045 0.122 0.045]
   [0.122 0.332 0.122]
   [0.045 0.122 0.045]]
  ```
- Sobel X Kernel:
  ```
  [[-1  0  1]
   [-2  0  2]
   [-1  0  1]]
  ```
- Sobel Y Kernel:
  ```
  [[ 1  2  1]
   [ 0  0  0]
   [-1 -2 -1]]
  ```

### Notes

- All image paths should be relative to the project directory
- Application uses OpenCV's default BGR color space
- Smoothing operations use odd-numbered kernel sizes (2m+1)
- Custom implementations avoid built-in OpenCV functions for edge detection

## Controls

- ESC: Close popup windows
- Trackbars: Adjust parameters for blending and filtering operations

## License

This project is available under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Based on OpenCV computer vision library
- Created as part of image processing coursework
