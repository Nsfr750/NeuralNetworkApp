from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, 
                             QDialogButtonBox, QTextBrowser)

# Local imports
from neuralnetworkapp.version import __version__


class AboutDialog(QDialog):
    """Custom about dialog with close button."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Neural Network Creator")
        self.setMinimumSize(400, 300)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create text browser for the about content
        about_text = f"""
        <h2 style='text-align: center;'>Neural Network Creator</h2>
        <p style='text-align: center;'>Version: {__version__}</p>
        <p style='text-align: center;'>© Copyright 2025 Nsfr750 - All rights reserved</p>
        <p style='text-align: justify;'>A PySide6 application for creating and managing neural networks.</p>
        <p style='text-align: center;'>GitHub: <a href='https://github.com/Nsfr750/NeuralNetworkApp'>Nsfr750/NeuralNetworkApp</a></p>
        """
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(about_text)
        text_browser.setReadOnly(True)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.close)
        
        # Add widgets to layout
        layout.addWidget(text_browser)
        layout.addWidget(button_box)
        
        # Set layout
        self.setLayout(layout)


def show_about(parent=None):
    """Show the about dialog.
    
    Args:
        parent: Parent widget for the dialog
    """
    dialog = AboutDialog(parent)
    dialog.setModal(True)
    dialog.exec()
