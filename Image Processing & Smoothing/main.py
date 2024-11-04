# Import required PyQt5 modules
from PyQt5 import QtWidgets  # Contains widget classes for creating GUI elements
from PyQt5 import QtGui      # Provides classes for windowing system integration
from PyQt5 import QtCore     # Core non-GUI functionality
from controller import MainWindow_controller  # Import our custom window controller class
import sys  # Required for system-level operations and command line arguments

# Main entry point of the application
if __name__ == '__main__':
    """
    Main entry point of the PyQt5 application.
    This block only executes if the script is run directly (not imported as a module).
    """
    
    # Create a QApplication instance
    # QApplication manages the GUI application's control flow and main settings
    # sys.argv passes command-line arguments to the application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create an instance of our main window controller
    # This creates the main application window with all UI elements
    window = MainWindow_controller()

    # Make the window visible
    # show() makes the window visible on screen
    # Unlike show(), showMaximized() would show the window in full screen
    window.show()
    
    # Start the application's event loop
    # app.exec_() starts Qt's event loop
    # sys.exit() ensures a clean exit when the event loop stops
    # The event loop handles all user interactions, window events, etc.
    sys.exit(app.exec_())