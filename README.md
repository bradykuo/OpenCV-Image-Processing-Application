# OpenCV Image Processing Application
A PyQt5-based application for demonstrating image processing techniques using OpenCV.
（成大資工系｜影像處理、電腦視覺及深度學習概論｜作業）

## Features
### 1. Image Processing 
- 1.1. Color Separation 
- 1.2. Color Transformation 
- 1.3. Color Detection 
- 1.4. Blending 

### 2. Image Smoothing 
- 2.1. Gaussian Blur 
- 2.2. Bilateral Filter 
- 2.3. Median Filter 

### 3. Edge Detection
- 3.1. Gaussian Blur
- 3.2. Sobel X (Vertical Edge Detection)
- 3.3. Sobel Y (Horizontal Edge Detection)
- 3.4. Magnitude

### 4. Transforms
- 4.1. Resize
- 4.2. Translation
- 4.3. Rotation & Scaling
- 4.4. Shearing

## Requirements
### Python Environment
- Python 3.7

### Core Dependencies
- opencv-contrib-python (3.4.2.17)
- Matplotlib 3.1.1
- PyQt5 (5.15.1)
- numpy (>=1.19.0)

### Additional Dependencies
- numpy
- pillow

### Required Images
#### Image Processing
- OpenCV.png (for color operations)
- Dog_Strong.jpg and Dog_Weak.jpg (for blending)
- beauty.png (for smoothing operations)
- building.jpg (for edge detection)
- Microsoft.png (for transforms)

## Installation and Setup
### General Setup
1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create virtual environment:
```bash
python -m venv venv
```

### Windows Setup
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/Mac Setup
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Image Processing Operations
#### Basic Operation
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

#### Feature-Specific Operations
##### Edge Detection
1. Load building.jpg using "Load Image 1" button
2. Select operations in sequence:
   - "3.1 Gaussian Blur" for initial smoothing
   - "3.2 Sobel X" for vertical edges
   - "3.3 Sobel Y" for horizontal edges
   - "3.4 Magnitude" for combined edge strength

##### Transform Operations
1. Load Microsoft.png using "Load Image 1" button
2. Apply transformations in sequence:
   - "4.1 Resize" (430x430 → 215x215, centered at (108, 108))
   - "4.2 Translation" (shift from (108, 108) to (323, 323))
   - "4.3 Rotation, Scaling" (45° counter-clockwise, 0.5x scale)
   - "4.4 Shearing" (points transform: [[50,50],[200,50],[50,200]] → [[10,100],[100,50],[100,250]])

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

#### Edge Magnitude Formula
```
Magnitude = sqrt(SobelX² + SobelY²)
Normalized to range [0, 255]
```

### Notes
- All image paths should be relative to the project directory
- Application uses OpenCV's default BGR color space
- Smoothing operations use odd-numbered kernel sizes (2m+1)
- Custom implementations avoid built-in OpenCV functions for edge detection

## Controls
- ESC: Close popup windows
- Trackbars: Adjust parameters for image processing operations

## Troubleshooting
- If image windows don't close properly, press any key and then 'ESC'
- For optimal performance, ensure input images match the recommended dimensions
- If trackbars aren't responding, click on the image window to ensure it's in focus
- Common dimension issues:
  - Edge detection expects images of sufficient resolution for clear edge detection
  - Transform operations are optimized for 430x430 input images
  - Resizing operations maintain aspect ratio by default

## Reference Formula
### Gaussian Blur
```
G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
where σ = √0.5
```

### Color Transformation
```
Grayscale conversion:
I1 = 0.07*B + 0.72*G + 0.21*R  (OpenCV function)
I2 = (R+G+B)/3                  (Average weighted)
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is available for academic and educational purposes.

## Acknowledgments
- Based on OpenCV computer vision library
- Created as part of image processing coursework
