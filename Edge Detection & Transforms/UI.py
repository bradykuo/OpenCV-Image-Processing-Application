# -*- coding: utf-8 -*-

# This file was automatically generated from a .ui file created in Qt Designer
# Warning: Manual changes to this file will be lost if the UI file is regenerated

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Set up the main window properties
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)  # Set window size
        
        # Configure main window font settings
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        
        # Create central widget (main container)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Create "Load Image" button
        self.loadImage1_B = QtWidgets.QPushButton(self.centralwidget)
        self.loadImage1_B.setGeometry(QtCore.QRect(70, 250, 151, 61))
        self.loadImage1_B.setObjectName("loadImage1_B")
        
        # Create first vertical layout widget (Edge Detection section)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(280, 140, 241, 311))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        
        # Add Edge Detection buttons
        self.gaussianBlur_B = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.sobelX_B = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.sobelY_B = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.blending_B = QtWidgets.QPushButton(self.verticalLayoutWidget)
        
        # Add buttons to vertical layout
        self.verticalLayout.addWidget(self.gaussianBlur_B)
        self.verticalLayout.addWidget(self.sobelX_B)
        self.verticalLayout.addWidget(self.sobelY_B)
        self.verticalLayout.addWidget(self.blending_B)
        
        # Create second vertical layout widget (Transformation section)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(540, 140, 241, 311))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        
        # Add Transformation buttons
        self.resize_b = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.translation_B = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.rotation_B = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.median_B_2 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        
        # Add buttons to second vertical layout
        self.verticalLayout_2.addWidget(self.resize_b)
        self.verticalLayout_2.addWidget(self.translation_B)
        self.verticalLayout_2.addWidget(self.rotation_B)
        self.verticalLayout_2.addWidget(self.median_B_2)
        
        # Create section labels
        self.tital3_L = QtWidgets.QLabel(self.centralwidget)
        self.tital3_L.setGeometry(QtCore.QRect(280, 110, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.tital3_L.setFont(font)
        
        self.title4_L = QtWidgets.QLabel(self.centralwidget)
        self.title4_L.setGeometry(QtCore.QRect(540, 110, 191, 31))
        self.title4_L.setFont(font)
        
        # Create label for displaying image path
        self.image1Path_L = QtWidgets.QLabel(self.centralwidget)
        self.image1Path_L.setGeometry(QtCore.QRect(50, 330, 191, 31))
        self.image1Path_L.setFont(QtGui.QFont("Arial", 9))
        
        # Set up menubar and statusbar
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # Connect signal/slots and set text for widgets
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """Set text for all widgets that display text"""
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HW1"))
        # Set text for all buttons
        self.loadImage1_B.setText(_translate("MainWindow", "Load image 1"))
        self.gaussianBlur_B.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.sobelX_B.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.sobelY_B.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.blending_B.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.resize_b.setText(_translate("MainWindow", "4.1 Resize"))
        self.translation_B.setText(_translate("MainWindow", "4.2 Translation"))
        self.rotation_B.setText(_translate("MainWindow", "4.3 Rotation,Scaling"))
        self.median_B_2.setText(_translate("MainWindow", "4.4 Shearing"))
        # Set text for labels
        self.tital3_L.setText(_translate("MainWindow", "3.Edge Detection"))
        self.title4_L.setText(_translate("MainWindow", "4.Transformation"))


if __name__ == "__main__":  # Only execute if file is run directly, not imported
    import sys  # Import system module for command line arguments and exit handling
    
    # Create Qt application instance
    # QApplication manages the application's control flow and main settings
    # sys.argv allows command line arguments to be passed to the application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create the main window instance
    # QMainWindow provides a framework for building application's user interface
    MainWindow = QtWidgets.QMainWindow()
    
    # Create an instance of the UI class we defined above
    ui = Ui_MainWindow()
    
    # Set up the UI by calling setupUi method
    # This method creates and arranges all the widgets we defined
    ui.setupUi(MainWindow)
    
    # Make the main window visible
    MainWindow.show()
    
    # Start the application's event loop
    # sys.exit ensures proper cleanup when the application is closed
    # app.exec_() starts Qt's event loop and returns an exit code when done
    sys.exit(app.exec_())
