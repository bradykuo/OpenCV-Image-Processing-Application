# Import required libraries
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QPushButton, QMessageBox
import cv2
import os
import glob
import math

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    """
    Main window controller class that handles application logic
    Inherits from QMainWindow
    """
    def __init__(self):
        # Initialize parent class
        super().__init__()
        
        # Create and set up the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Connect signals to slots
        self.setup_control()
        
        # Initialize global variables for image handling
        global filename1, image1
        filename1 = None  # Stores the path to the image
        image1 = None    # Stores the image data

    def resize(self, width, height):
        """Handle window resize events"""
        super().resize(width, height)

    def setup_control(self):
        """Connect button clicks to their respective functions"""
        # Connect each button's clicked signal to its corresponding method
        self.ui.loadImage1_B.clicked.connect(self.load_img1)
        self.ui.gaussianBlur_B.clicked.connect(self.gaussianBlur)
        self.ui.sobelX_B.clicked.connect(self.sobelX)
        self.ui.sobelY_B.clicked.connect(self.sobelY)
        self.ui.blending_B.clicked.connect(self.magnitude)
        self.ui.resize_b.clicked.connect(self.resize_image)
        self.ui.translation_B.clicked.connect(self.translation)
        self.ui.rotation_B.clicked.connect(self.rotation)
        self.ui.median_B_2.clicked.connect(self.shearing)

    def load_img1(self):
        """Handle image loading via file dialog"""
        global filename1, image1
        # Open file dialog and get image path
        filename1, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        print(filename1, filetype)
        # Display the image path in the UI
        self.ui.image1Path_L.setText(filename1)
        image1 = filename1

    def gaussianBlur(self):
        """Apply Gaussian blur to the loaded image"""
        image = cv2.imread(filename1)

        def rgb_to_gray(rgb_image):
            """Convert RGB image to grayscale using standard coefficients"""
            return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
        
        def generate_gaussian_kernel():
            """Generate 3x3 Gaussian kernel with σ = √0.5"""
            sigma = np.sqrt(0.5)
            kernel = np.zeros((3, 3))
            
            # Calculate Gaussian values for each position
            for x in range(-1, 2):
                for y in range(-1, 2):
                    kernel[x+1, y+1] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
            
            # Normalize kernel
            return kernel / np.sum(kernel)
        
        def apply_gaussian_filter(gray_image, kernel):
            """Apply convolution with Gaussian kernel"""
            height, width = gray_image.shape
            output = np.zeros_like(gray_image)
            
            # Add padding for border handling
            padded = np.pad(gray_image, ((1, 1), (1, 1)), mode='edge')
            
            # Apply convolution
            for i in range(height):
                for j in range(width):
                    window = padded[i:i+3, j:j+3]
                    output[i, j] = np.sum(window * kernel)
                    
            return output
        
        # Process image
        gray_image = rgb_to_gray(image)
        kernel = generate_gaussian_kernel()
        blurred = apply_gaussian_filter(gray_image, kernel)
        blurred = blurred.astype(np.uint8)
        
        # Display result
        cv2.imshow("Gaussian Blur", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def sobelX(self):
        """Apply Sobel X edge detection to the image"""
        image = cv2.imread(filename1)
    
        def rgb_to_gray(rgb_image):
            """Convert RGB to grayscale"""
            return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
        
        def generate_gaussian_kernel():
            """Generate Gaussian kernel for pre-processing"""
            sigma = np.sqrt(0.5)
            kernel = np.zeros((3, 3))
            for x in range(-1, 2):
                for y in range(-1, 2):
                    kernel[x+1, y+1] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
            return kernel / np.sum(kernel)
        
        def apply_filter(image, kernel):
            """Apply convolution filter (used for both Gaussian and Sobel)"""
            height, width = image.shape
            output = np.zeros_like(image, dtype=np.float32)
            
            padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
            
            for i in range(height):
                for j in range(width):
                    window = padded[i:i+3, j:j+3]
                    output[i, j] = np.sum(window * kernel)
                    
            return output
        
        # Process image
        gray_image = rgb_to_gray(image)
        gaussian_kernel = generate_gaussian_kernel()
        blurred = apply_filter(gray_image, gaussian_kernel)
        
        # Define Sobel X kernel for vertical edge detection
        sobel_x_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        
        # Apply Sobel X operator and normalize
        edge_x = apply_filter(blurred, sobel_x_kernel)
        edge_x = np.abs(edge_x)
        edge_x = (edge_x * 255.0 / edge_x.max()).astype(np.uint8)
        
        # Display results
        cv2.imshow("Sobel X Edge Detection", edge_x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def sobelY(self):
        """Implementation of Sobel Y operator for horizontal edge detection"""
        image = cv2.imread(filename1)
        
        def rgb_to_gray(rgb_image):
            """Convert RGB image to grayscale using weighted sum"""
            return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
        
        def generate_gaussian_kernel():
            """Generate 3x3 Gaussian kernel with σ = √0.5 for noise reduction"""
            sigma = np.sqrt(0.5)
            kernel = np.zeros((3, 3))
            for x in range(-1, 2):
                for y in range(-1, 2):
                    kernel[x+1, y+1] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
            return kernel / np.sum(kernel)
        
        def apply_filter(image, kernel):
            """Apply convolution filter to image using provided kernel"""
            height, width = image.shape
            output = np.zeros_like(image, dtype=np.float32)
            padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
            
            for i in range(height):
                for j in range(width):
                    window = padded[i:i+3, j:j+3]
                    output[i, j] = np.sum(window * kernel)
            return output
        
        # Process image through pipeline
        gray_image = rgb_to_gray(image)
        blurred = apply_filter(gray_image, generate_gaussian_kernel())
        
        # Define Sobel Y kernel for horizontal edge detection
        sobel_y_kernel = np.array([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1]
        ])
        
        # Apply Sobel Y and normalize
        edge_y = apply_filter(blurred, sobel_y_kernel)
        edge_y = np.abs(edge_y)
        edge_y = (edge_y * 255.0 / edge_y.max()).astype(np.uint8)
        
        # Display result
        cv2.imshow("Sobel Y Edge Detection", edge_y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def magnitude(self):
        """Calculate edge magnitude from Sobel X and Y operators"""
        image = cv2.imread(filename1)
       
        # [Previous helper functions: rgb_to_gray, generate_gaussian_kernel, apply_filter]
        # Same implementations as in sobelY()

        # Convert RGB to grayscale
        def rgb_to_gray(rgb_image):
            return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
        
        # Generate Gaussian kernel
        def generate_gaussian_kernel():
            sigma = np.sqrt(0.5)
            kernel = np.zeros((3, 3))
            for x in range(-1, 2):
                for y in range(-1, 2):
                    kernel[x+1, y+1] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
            return kernel / np.sum(kernel)
        
        # Apply convolution (used for both Gaussian and Sobel)
        def apply_filter(image, kernel):
            height, width = image.shape
            output = np.zeros_like(image, dtype=np.float32)
            
            # Add padding
            padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
            
            # Apply convolution
            for i in range(height):
                for j in range(width):
                    window = padded[i:i+3, j:j+3]
                    output[i, j] = np.sum(window * kernel)
                        
            return output
        
        # Process image through initial pipeline
        gray_image = rgb_to_gray(image)
        blurred = apply_filter(gray_image, generate_gaussian_kernel())
        
        # Define both Sobel kernels
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Vertical edges
        sobel_y_kernel = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]])  # Horizontal edges
        
        # Apply both Sobel operators
        edge_x = apply_filter(blurred, sobel_x_kernel)
        edge_y = apply_filter(blurred, sobel_y_kernel)
        
        # Calculate magnitude using Pythagorean theorem
        magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
        magnitude = (magnitude * 255.0 / magnitude.max()).astype(np.uint8)
        
        # Display results
        cv2.imshow("Edge Magnitude", magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def resize_image(self):
        """Resize and combine images with binary threshold"""
        image1 = cv2.imread(filename1)
        
        # Resize original image to 215x215
        img1 = cv2.resize(image1, (215, 215))
        
        # Create binary threshold version and resize to 430x430
        ret, image2 = cv2.threshold(image1, 255, 255, cv2.THRESH_BINARY)   
        img2 = cv2.resize(image2, (430, 430))
        
        # Overlay smaller image onto larger one at position (10,10)
        x, y = 10, 10
        W1, H1 = img2.shape[1::-1]  # Get dimensions
        W2, H2 = img1.shape[1::-1]
        
        # Crop region and add images
        imgCrop = img2[y:y+H2, x:x+W2]
        imgAdd = cv2.add(imgCrop, img1)
        
        # Create final composite
        imgAddM = np.array(img2)
        imgAddM[y:y+H2, x:x+W2] = imgAdd
        
        cv2.imshow('imgAddM', imgAddM)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # function for translation
    def translation(self):
        """Apply translation transformation to the image"""
        image1 = cv2.imread(filename1)
        
        # Initial resize and threshold operations (same as resize_image)
        img1 = cv2.resize(image1, (215, 215))
        ret, image2 = cv2.threshold(image1, 255, 255, cv2.THRESH_BINARY)   
        img2 = cv2.resize(image2, (430, 430))
        
        # First overlay at (10,10)
        x,y = 10,10
        W1,H1 = img2.shape[1::-1]
        W2,H2 = img1.shape[1::-1]
        
        imgCrop = img2[y:y+H2,x:x+W2]
        imgAdd = cv2.add(imgCrop,img1)
        
        imgAddM = np.array(img2)
        imgAddM[y:y+H2,x:x+W2] = imgAdd
        
        # Second overlay at (210,210)
        x,y = 210,210
        W1,H1 = imgAddM.shape[1::-1]
        W2,H2 = img1.shape[1::-1]
        
        imgCrop = imgAddM[y:y+H2,x:x+W2]
        imgAdd = cv2.add(imgCrop,img1)
        
        imgAddM2 = np.array(imgAddM)
        imgAddM2[y:y+H2,x:x+W2] = imgAdd
        
        cv2.imshow('imgAddM',imgAddM2)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    # function for rotation
    def rotation(self):
        """Apply rotation transformation to the image"""
        image1 = cv2.imread(filename1)

        # Initial processing (same as translation)
        
        img1 = cv2.resize(image1, (215, 215))
        ret, image2 = cv2.threshold(image1, 255, 255, cv2.THRESH_BINARY)   
        img2 = cv2.resize(image2, (430, 430))
        
        x,y = 10,10
        W1,H1 = img2.shape[1::-1]
        W2,H2 = img1.shape[1::-1]
        
        imgCrop = img2[y:y+H2,x:x+W2]
        imgAdd = cv2.add(imgCrop,img1)
        
        imgAddM = np.array(img2)
        imgAddM[y:y+H2,x:x+W2] = imgAdd
        
        x,y = 210,210
        W1,H1 = imgAddM.shape[1::-1]
        W2,H2 = img1.shape[1::-1]
        
        imgCrop = imgAddM[y:y+H2,x:x+W2]
        imgAdd = cv2.add(imgCrop,img1)

        imgAddM2 = np.array(imgAddM)
        imgAddM2[y:y+H2,x:x+W2] = imgAdd

        # Additional processing for rotation
        imgAddM3 = cv2.resize(imgAddM2, (0, 0), fx=0.5, fy=0.5)
        # Final overlay
        x,y = 108,108
        W1,H1 = img2.shape[1::-1]
        W2,H2 = imgAddM3.shape[1::-1]
        
        imgCrop = img2[y:y+H2,x:x+W2]
        imgAdd = cv2.add(imgCrop,imgAddM3)
        
        img3 = np.array(img2)
        img3[y:y+H2,x:x+W2] = imgAdd
        
        # Apply 45-degree rotation
        rows,cols = 430,430
        
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),45,1)
        dst = cv2.warpAffine(img3,M,(cols,rows))
        
        cv2.imshow('dst',dst)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
    def shearing(self):
        """
        Apply shearing transformation to image with multiple preprocessing steps
        """
        # Load and prepare initial images
        image1 = cv2.imread(filename1)
        
        # Create two versions of the image:
        # 1. Smaller version (215x215)
        img1 = cv2.resize(image1, (215, 215))
        # 2. Binary thresholded version (430x430)
        ret, image2 = cv2.threshold(image1, 255, 255, cv2.THRESH_BINARY)   
        img2 = cv2.resize(image2, (430, 430))
        
        # First overlay: Position at (10,10)
        x, y = 10, 10
        # Get dimensions (width and height) of both images
        W1, H1 = img2.shape[1::-1]  # Dimensions of larger image
        W2, H2 = img1.shape[1::-1]  # Dimensions of smaller image
        
        # Overlay first image
        imgCrop = img2[y:y+H2, x:x+W2]  # Crop region from background
        imgAdd = cv2.add(imgCrop, img1)  # Add images in overlapping region
        
        # Create copy of background and insert overlay
        imgAddM = np.array(img2)
        imgAddM[y:y+H2, x:x+W2] = imgAdd
        
        # Second overlay: Position at (210,210)
        x, y = 210, 210
        W1, H1 = imgAddM.shape[1::-1]
        W2, H2 = img1.shape[1::-1]
        
        # Repeat overlay process
        imgCrop = imgAddM[y:y+H2, x:x+W2]
        imgAdd = cv2.add(imgCrop, img1)
        imgAddM2 = np.array(imgAddM)
        imgAddM2[y:y+H2, x:x+W2] = imgAdd
        
        # Third overlay: Scale down and position at (108,108)
        imgAddM3 = cv2.resize(imgAddM2, (0, 0), fx=0.5, fy=0.5)  # Scale by 50%
        x, y = 108, 108
        W1, H1 = img2.shape[1::-1]
        W2, H2 = imgAddM3.shape[1::-1]
        
        # Perform third overlay
        imgCrop = img2[y:y+H2, x:x+W2]
        imgAdd = cv2.add(imgCrop, imgAddM3)
        img3 = np.array(img2)
        img3[y:y+H2, x:x+W2] = imgAdd
        
        # Apply 45-degree rotation
        rows, cols = 430, 430
        M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 45, 1)
        dst = cv2.warpAffine(img3, M, (cols, rows))
        
        # Apply shearing transformation
        # Define source points
        p1 = np.float32([
            [50, 50],    # Top-left point
            [200, 50],   # Top-right point
            [50, 200]    # Bottom-left point
        ])
        # Define destination points for shearing
        p2 = np.float32([
            [10, 100],   # New top-left
            [100, 50],   # New top-right
            [100, 250]   # New bottom-left
        ])
        
        # Calculate and apply affine transformation
        M = cv2.getAffineTransform(p1, p2)  # Get transformation matrix
        output = cv2.warpAffine(dst, M, (430, 430))  # Apply transformation
        
        # Display result
        cv2.imshow('output', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)