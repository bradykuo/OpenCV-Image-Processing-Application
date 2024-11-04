from PyQt5 import QtWidgets, QtGui, QtCore  # Import the main PyQt5 modules
from controller import MainWindow_controller  # Import the custom controller class
import sys  # Import system-specific parameters and functions

if __name__ == '__main__':  # Ensure this code only runs if this file is run directly
    # Create a QApplication instance
    # QApplication manages the GUI application's control flow and main settings
    # sys.argv passes command-line arguments to the application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create an instance of our main window controller
    # This will set up the main application window and its functionality
    window = MainWindow_controller()

    # Make the window visible
    window.show()
    
    # Start the application's event loop and handle clean exit
    # app.exec_() starts the event loop
    # sys.exit() ensures a clean exit when the event loop is terminated
    sys.exit(app.exec_())