# OpenCV Image Processing & Deep Learning Application
A PyQt5-based application for demonstrating image processing techniques using OpenCV and deep learning classification using VGG19 on CIFAR10 dataset.

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

### 5. CIFAR10 Classification using VGG19
- 5.1. Display Training Images
- 5.2. Model Structure Visualization
- 5.3. Data Augmentation Demonstration
- 5.4. Training Progress Visualization
- 5.5. Image Classification Inference

## Requirements

### Python Environment
- Python 3.7

### Core Dependencies
- opencv-contrib-python (3.4.2.17)
- Matplotlib 3.1.1
- PyQt5 (5.15.1)

### Deep Learning Dependencies
- PyTorch
- torchvision
- torchsummary

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

#### Classification
- Any test images for inference (supports .jpg, .jpeg, .png)

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

### 1. Image Processing Operations

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
   - "4.1 Resize" (430x430 → 215x215)
   - "4.2 Translation" (+215 pixels X/Y)
   - "4.3 Rotation, Scaling" (45°, 0.5x)
   - "4.4 Shearing" (affine transform)

### 2. CIFAR10 Classification

#### Training the Model
1. Run the training script:
```bash
python train.py
```
- Trains VGG19 model for 30 epochs
- Automatically saves:
  - Model checkpoints
  - Training progress
  - Performance plots

#### Using Classification Features
1. Launch the application:
```bash
python main.py
```

2. Use the interface:
   - "1. Show Train Images": View CIFAR10 samples
   - "2. Show Model Structure": VGG19 architecture
   - "3. Show Data Augmentation": Augmentation examples
   - "4. Show Accuracy and Loss": Training progress
   - "5. Inference": Classify new images

## Technical Details

### Project Structure
```
project/
│
├── main.py           # Main application entry point
├── train.py          # VGG19 training script
├── controller.py     # Main logic and image processing functions
├── UI.py            # PyQt5 UI definition
│
├── models/          # Saved model weights
│   └── vgg19_final.pth
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

#### 1. Image Processing
##### Edge Detection Kernels
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

#### 2. Deep Learning
- Model Architecture: VGG19
  - Pretrained on ImageNet
  - Modified for CIFAR10 (10 classes)
  - Input Size: 32x32x3

- Training Configuration:
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
  - Epochs: 30
  - Batch Size: 32

- Data Augmentation:
  - Random Rotation
  - Random Resized Crop
  - Random Horizontal Flip

### Notes
- All image paths should be relative to the project directory
- Application uses OpenCV's default BGR color space
- Smoothing operations use odd-numbered kernel sizes (2m+1)
- Custom implementations avoid built-in OpenCV functions for edge detection
- Model training time: ~8-12 hours on CPU, ~1-2 hours on GPU

## Reference

https://reurl.cc/MjaKbk

## Controls
- ESC: Close popup windows
- Trackbars: Adjust parameters for image processing operations

## License
This project is available under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- Based on OpenCV computer vision library
- Uses PyTorch deep learning framework
- CIFAR10 dataset provided by the Canadian Institute For Advanced Research
- Created as part of image processing and deep learning coursework
