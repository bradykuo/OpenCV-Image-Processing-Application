# OpenCV Image Processing Application

A PyQt5-based application for demonstrating various image processing techniques using OpenCV. This project implements basic image processing operations including color manipulation, image blending, and different types of image smoothing filters.

## Features

### 1. Image Processing 
1.1. Color Separation 
- Extracts and displays BGR channels separately
- Uses OpenCV.png as input
- Shows individual Blue, Green, and Red channel visualizations

1.2. Color Transformation 
- Implements two methods of grayscale conversion:
  - OpenCV's built-in weighted conversion (I1 = 0.07*B + 0.72*G + 0.21*R)
  - Simple averaging method (I2 = (R+G+B)/3)
- Uses OpenCV.png as input

1.3. Color Detection 
- Detects specific colors (Green and White) using HSV color space
- Color ranges:
  - Green: H(40-80), S(50-255), V(20-255)
  - White: H(0-180), S(0-20), V(200-255)
- Uses OpenCV.png as input

1.4. Blending 
- Interactive image blending between two images
- Features a trackbar for adjusting blend weights
- Uses Dog_Strong.jpg and Dog_Weak.jpg

### 2. Image Smoothing 
2.1. Gaussian Blur 
- Interactive Gaussian blur with adjustable kernel size
- Magnitude range: 0-10 (kernel size = 2m+1)
- Uses beauty.png as input

2.2. Bilateral Filter 
- Edge-preserving smoothing filter
- Adjustable filter parameters with trackbar
- Uses beauty.png as input

2.3. Median Filter 
- Noise reduction using median filtering
- Adjustable kernel size via trackbar
- Uses beauty.png as input

## Requirements

### Python Dependencies
- Python 3.7
- opencv-contrib-python (3.4.2.17)
- Matplotlib 3.1.1
- PyQt5 (5.15.1)
- NumPy

### Installation
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install opencv-contrib-python==3.4.2.17
pip install matplotlib==3.1.1
pip install PyQt5==5.15.1
pip install numpy
```

### Required Images
Place the following images in your project directory:
- OpenCV.png (for color operations)
- Dog_Strong.jpg (for blending)
- Dog_Weak.jpg (for blending)
- beauty.png (for smoothing operations)

## Usage

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

4. Using the Interface:
   - Use "Load Image 1" for single image operations
   - Use both "Load Image 1" and "Load Image 2" for blending
   - Click on operation buttons to perform specific image processing tasks
   - Use trackbars in popup windows to adjust parameters
   - Press 'Esc' to close popup windows

## Project Structure
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
│   └── beauty.png
│
└── requirements.txt  # Project dependencies
```

## Controls
- ESC: Close popup windows
- Trackbars: Adjust parameters for blending and filtering operations

## Notes
- All image paths should be relative to the project directory
- The application uses OpenCV's default BGR color space
- Smoothing operations use odd-numbered kernel sizes (2m+1)

## License
[Your chosen license]

## Acknowledgments
- Based on OpenCV computer vision library
- Created as part of image processing coursework
