import numpy as np
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QPushButton, QMessageBox
import cv2
import os
import glob

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    #here is the button controller
    def setup_control(self):
        self.ui.load_img1_ui.clicked.connect(self.load_img1)
        self.ui.load_img2_ui.clicked.connect(self.load_img2)
        #two following code were use to switch to another scene and then display picture input
        #self.ui.color_seperation_ui.clicked.connect(self.switchto_function1)
        #self.ui.color_seperation_ui.clicked.connect(self.displayImage)
        self.ui.color_seperation_ui.clicked.connect(self.color_separation)
        self.ui.color_transformation_ui.clicked.connect(self.color_transformation)
        self.ui.color_detection_ui.clicked.connect(self.color_detection)
        self.ui.blending_ui.clicked.connect(self.blending)
        self.ui.Gaussian_blur_ui.clicked.connect(self.Gaussian_blur)
        self.ui.Bilateral_filter_ui.clicked.connect(self.Bilateral_filter)
        self.ui.Median_filter_ui.clicked.connect(self.Median_filter)


    #function for uploading images 
    def load_img1(self):
        global filename1
        global image1
        filename1, filetype = QFileDialog.getOpenFileName(self,"Open file","./")
        print(filename1, filetype)
        self.ui.img1_path.setText(filename1)
        image1 = filename1

    def load_img2(self):
        global filename2
        global image2
        filename2, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        print(filename2, filetype)
        self.ui.img2_path.setText(filename2)
        image2 = filename2

    #function for color seperation
    def color_separation(self):
        img = cv2.imread(filename1)
        zero_channel = np.zeros(img.shape[0:2], dtype = "uint8")
        
        #拆成R,B,G三通道
        B, G, R = cv2.split(img) 
        
        #通道分量為0，視為零矩陣
        img_B = cv2.merge([B, zero_channel, zero_channel])
        img_G = cv2.merge([zero_channel, G, zero_channel])
        img_R = cv2.merge([zero_channel, zero_channel, R])
        
        cv2.imshow("B", img_B)
        cv2.imshow("G", img_G)
        cv2.imshow("R", img_R)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #press 1 or 0 to quit

    #funtion for color transform
    def color_transformation(self):
        img = cv2.imread(filename1)
        
        #OpenCV function
        #Perceptually weighted formula: I1 = 0.07*B + 0.72*G + 0.21*R
        I1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('I1 (OpenCV function)',I1)
        
        #Average weighted
        #Average weighted formula: I2 = (R+G+B)/3
        img[:] = np.sum(img, axis = -1, keepdims = 1)/3
        
        cv2.imshow('I2 (Average weighted)',img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    #function for color detection
    def color_detection(self):
        img = cv2.imread(filename1)
        
        #Green Range : (40-80,50-255,20-255)
        hsv_g = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 20])
        upper_green = np.array([80, 255, 255])
        
        #White Range : (0-180,0-20,200-255)
        hsv_w = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 20, 255])
        
        #使用遮罩
        mask1 = cv2.inRange(hsv_g, lower_green, upper_green)
        res = cv2.bitwise_and(img, img, mask=mask1)
        cv2.imshow('green', res)
        
        mask2 = cv2.inRange(hsv_w, lower_white, upper_white)
        res = cv2.bitwise_and(img, img, mask=mask2)
        cv2.imshow('white', res)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    #function for blending image
    def blending(self):
        def nothing(x):
            pass

        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)
        
        #建立一個黑色背景的視窗
        img = np.zeros((500,500,3), np.uint8)
        cv2.namedWindow('image')
        
        cv2.createTrackbar("Blending", "image", 0, 255, nothing)
        cv2.setTrackbarPos("Blending", "image", 128)
        
        while(1):
            cv2.imshow('image',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            
            r = cv2.getTrackbarPos("Blending", "image")
            r = float(r)/255.0
            
            #隨Trackbar變動而改變
            img = cv2.addWeighted(img1,r,img2,1.0-r,0)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    #function for making Gaussian blur
    def Gaussian_blur(self):
        # img 來源影像
        # ksize 指定區域單位 ( 必須是大於 1 的奇數 )
        # sigmaX X 方向標準差，預設 0，sigmaY Y 方向標準差，預設 0
        def nothing(x):
            pass
        
        img1 = cv2.imread(filename1)
        
        img = np.zeros((500,500,3), np.uint8)
        cv2.namedWindow('image')
        
        cv2.createTrackbar("Magnitude", "image", 0, 10, nothing)
        cv2.setTrackbarPos("Magnitude", "image", 5)
        
        while(1):
            cv2.imshow('image',img)
            x = cv2.waitKey(1) & 0xFF
            if x == 27:
                break
            
            r = cv2.getTrackbarPos("Magnitude", "image")
            k = 2 * r + 1 
            r = float(r)/10.0
            
            # 指定區域單位為 (k, k) 
            # k = 2m+1
            img = cv2.GaussianBlur(img1, (k, k), 0)   
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # function for making Bilateral filter
    def Bilateral_filter(self):
        # img 來源影像
        # d 相鄰像素的直徑，預設使用 5，數值越大運算的速度越慢
        # sigmaColor 相鄰像素的顏色混合，數值越大，會混合更多區域的顏色，並產生更大區塊的同一種顏色
        # sigmaSpace 會影響像素的區域，數值越大，影響的範圍就越大，影響的像素就越多
        def nothing(x):
            pass
        
        img1 = cv2.imread(filename1)
        
        img = np.zeros((500,500,3), np.uint8)
        cv2.namedWindow('image')
        
        cv2.createTrackbar("Magnitude", "image", 0, 10, nothing)
        cv2.setTrackbarPos("Magnitude", "image", 5)
        
        while(1):
            cv2.imshow('image',img)
            x = cv2.waitKey(1) & 0xFF
            if x == 27:
                break
            
            r = cv2.getTrackbarPos("Magnitude", "image")
            k = 2 * r + 1 
            r = float(r)/10.0
            
            img = cv2.bilateralFilter(img1, k, 90, 90)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # function for making Median filter
    def Median_filter(self):
        # img 來源影像
        # ksize 模糊程度 ( 必須是大於 1 的奇數 )
        def nothing(x):
            pass
        
        img1 = cv2.imread(filename1)
        
        img = np.zeros((500,500,3), np.uint8)
        cv2.namedWindow('image')
        
        cv2.createTrackbar("Magnitude", "image", 0, 10, nothing)
        cv2.setTrackbarPos("Magnitude", "image", 5)
        
        while(1):
            cv2.imshow('image',img)
            x = cv2.waitKey(1) & 0xFF
            if x == 27:
                break
            
            r = cv2.getTrackbarPos("Magnitude", "image")
            k = 2 * r + 1 
            r = float(r)/10.0
            
            img = cv2.medianBlur(img1, k) # 模糊程度為 k
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
