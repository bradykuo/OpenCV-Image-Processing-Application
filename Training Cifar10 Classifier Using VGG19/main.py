import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchsummary import summary
from PIL import Image
import cv2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Set up the main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)  # Increased width for side-by-side layout
        MainWindow.setStyleSheet("""
            QMainWindow {
                background-color: #2B2B2B;
            }
            QWidget {
                background-color: #2B2B2B;
                color: white;
            }
            QPushButton {
                background-color: #3B3B3B;
                border: 2px solid red;
                color: white;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4B4B4B;
            }
            QLabel {
                color: white;
                font-size: 24px;
            }
        """)

        # Create central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        
        # Create main horizontal layout
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # Create left panel for buttons
        self.left_panel = QtWidgets.QVBoxLayout()
        self.left_panel.setSpacing(20)
        
        # Add title
        self.title_label = QtWidgets.QLabel("5. ResNet101 Test")
        self.left_panel.addWidget(self.title_label)
        
        # Add Load Image button
        self.load_image_btn = QtWidgets.QPushButton("Load Image")
        self.load_image_btn.setMinimumHeight(60)
        self.load_image_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #666;
                background-color: #3B3B3B;
            }
        """)
        self.left_panel.addWidget(self.load_image_btn)
        
        # Create frame for function buttons
        self.button_frame = QtWidgets.QFrame()
        self.button_frame.setStyleSheet("""
            QFrame {
                border: 2px solid red;
                background-color: transparent;
                padding: 20px;
            }
        """)
        self.button_layout = QtWidgets.QVBoxLayout(self.button_frame)
        self.button_layout.setSpacing(20)
        
        # Create function buttons
        self.buttons = {}
        button_texts = [
            "1. Show Train Images",
            "2. Show Model Structure",
            "3. Show Data Augmentation",
            "4. Show Accuracy and Loss",
            "5. Inference"
        ]
        
        for text in button_texts:
            button = QtWidgets.QPushButton(text)
            button.setMinimumHeight(60)
            button.setStyleSheet("""
                QPushButton {
                    border: 2px solid red;
                    background-color: #2B2B2B;
                    color: white;
                    text-align: center;
                    padding: 15px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #3B3B3B;
                }
            """)
            self.button_layout.addWidget(button)
            self.buttons[text] = button
        
        self.left_panel.addWidget(self.button_frame)
        
        # Add stretch to push everything to the top
        self.left_panel.addStretch()
        
        # Create right panel for image display
        self.right_panel = QtWidgets.QVBoxLayout()
        
        # Add image display area
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(600, 600)  # Increased size for better visibility
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #3B3B3B;
                border: 1px solid #666;
            }
        """)
        self.right_panel.addWidget(self.image_label)
        
        # Add layouts to main horizontal layout
        self.main_layout.addLayout(self.left_panel, 1)  # 1 is the stretch factor
        self.main_layout.addLayout(self.right_panel, 2)  # 2 is the stretch factor (larger)
        
        # Set central widget
        MainWindow.setCentralWidget(self.centralwidget)

        # Connect buttons to functions
        self.load_image_btn.clicked.connect(self.load_image)
        self.buttons["1. Show Train Images"].clicked.connect(self.show_train_images)
        self.buttons["2. Show Model Structure"].clicked.connect(self.show_model_structure)
        self.buttons["3. Show Data Augmentation"].clicked.connect(self.show_augmentation)
        self.buttons["4. Show Accuracy and Loss"].clicked.connect(self.show_accuracy_loss)
        self.buttons["5. Inference"].clicked.connect(self.run_inference)
        
        # Initialize data and model
        self.init_data()
        self.init_model()

    def init_data(self):
        """Initialize CIFAR10 data"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

    def init_model(self):
        """Initialize VGG19 model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vgg19(pretrained=True)
        # Modify the classifier for CIFAR10 (10 classes)
        self.model.classifier[6] = torch.nn.Linear(4096, 10)
        self.model = self.model.to(self.device)

    def show_train_images(self):
        """Show 9 random training images with labels"""
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(9):
            idx = torch.randint(len(self.trainset), size=(1,)).item()
            img, label = self.trainset[idx]
            img = img.numpy().transpose(1, 2, 0)  # Convert from tensor to numpy
            img = (img * 0.5 + 0.5)  # Unnormalize
            ax = axes[i//3, i%3]
            ax.imshow(img)
            ax.set_title(f'Label: {self.class_names[label]}')
            ax.axis('off')
        plt.tight_layout()
        self.show_matplotlib_figure(fig)

    def show_model_structure(self):
        """Display model structure"""
        summary(self.model, (3, 32, 32))

    def show_augmentation(self):
        """Show data augmentation results"""
        if not hasattr(self, 'current_image'):
            QtWidgets.QMessageBox.warning(None, "Warning", 
                                        "Please load an image first!")
            return

        try:
            # Get original image size
            original_height, original_width = self.current_image.shape[:2]
            
            # Define transformations with specific parameters according to documentation
            transforms_list = [
                # RandomRotation with interpolation
                transforms.RandomRotation(
                    degrees=30,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    center=None,
                    fill=0
                ),
                # RandomResizedCrop with standard parameters
                transforms.RandomResizedCrop(
                    size=(original_height, original_width),
                    scale=(0.08, 1.0),  # Default scale as per documentation
                    ratio=(0.75, 1.3333333333333333),  # Default ratio as per documentation
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True
                ),
                # RandomHorizontalFlip with default probability
                transforms.RandomHorizontalFlip(p=0.5)  # Default probability
            ]
            
            # Convert numpy image to PIL
            img_pil = Image.fromarray(self.current_image)
            
            # Create a figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # List of function names with parameters
            function_names = [
                "RandomRotation(30°)",
                f"RandomResizedCrop({original_height}x{original_width})",
                "RandomHorizontalFlip(p=0.5)"
            ]
            
            # Apply transformations
            for i, (transform, name) in enumerate(zip(transforms_list, function_names)):
                img_transformed = transform(img_pil)
                if isinstance(img_transformed, torch.Tensor):
                    img_transformed = img_transformed.numpy().transpose(1, 2, 0)
                else:
                    img_transformed = np.array(img_transformed)
                
                axes[i].imshow(img_transformed)
                axes[i].set_title(name)
                axes[i].axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to image and display
            canvas = FigureCanvas(fig)
            canvas.draw()
            
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(int(height), int(width), 3)
            
            # Display in main image label
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qt_image = QtGui.QImage(image.data, width, height, bytes_per_line, 
                                QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio))
            
            plt.close(fig)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Error", 
                                        f"Augmentation failed: {str(e)}")

    def show_accuracy_loss(self):
        """Show training accuracy and loss plots"""
        try:
            # Load the saved training history image directly
            training_history_path = 'training_history.png'
            
            if not os.path.exists(training_history_path):
                QtWidgets.QMessageBox.warning(None, "Error", 
                                            "Training history image not found!")
                return
            
            # Read the image
            img = cv2.imread(training_history_path)
            if img is None:
                QtWidgets.QMessageBox.warning(None, "Error", 
                                            "Could not read training history image!")
                return
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display in the main image label
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qt_image = QtGui.QImage(img.data, width, height, bytes_per_line, 
                                    QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            
            # Scale to fit the label while maintaining aspect ratio
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio))
                
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Error", 
                                        f"Could not display training history: {str(e)}")

    def run_inference(self):
        """Run inference on loaded image"""
        if not hasattr(self, 'current_image'):
            QtWidgets.QMessageBox.warning(None, "Warning", 
                                        "Please load an image first!")
            return
        
        try:
            # Prepare image for inference
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            img_pil = Image.fromarray(self.current_image)
            img_tensor = transform(img_pil).unsqueeze(0).to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            predicted_class = self.class_names[predicted.item()]
            confidence = confidence.item() * 100
            
            # Create figure to display results
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(self.current_image)
            ax.axis('off')
            plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%', 
                     pad=20)
            
            self.show_matplotlib_figure(fig)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "Error", 
                                        f"Inference failed: {str(e)}")

    def load_image(self):
        """Load and preprocess image from file"""
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_image)

    def display_image(self, image):
        """Display image in GUI"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(image.data, width, height, bytes_per_line, 
                               QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), QtCore.Qt.KeepAspectRatio))

    def show_matplotlib_figure(self, figure):
        """Display matplotlib figure in separate window"""
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Result")
        layout = QtWidgets.QVBoxLayout()
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        dialog.exec_()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())