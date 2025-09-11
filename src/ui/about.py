import os
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
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
        
        # Create horizontal layout for logo and content
        header_layout = QHBoxLayout()
        
        # Add logo to the left
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'logo.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        # Create text browser for the about content
        about_text = f"""
        <h2 style='text-align: center;'>Neural Network Creator</h2>
        <p style='text-align: center;'>Version: {__version__}</p>
        <p style='text-align: center;'>© Copyright 2025 Nsfr750 - All rights reserved</p>
        <p style='text-align: justify;'>A application for creating and managing neural networks.</p>
        <p style='text-align: center;'>GitHub: <a href='https://github.com/Nsfr750/NeuralNetworkApp'>Neural Network Creator</a></p>
        """
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(about_text)
        text_browser.setReadOnly(True)
        
        # Add close button with blue background and white text
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_button = button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_button:
            close_button.setStyleSheet("background-color: #0078d4; color: white; padding: 5px 15px; border: none; border-radius: 3px;")
        button_box.rejected.connect(self.close)
        
        # Add widgets to header layout
        header_layout.addWidget(logo_label)
        header_layout.addWidget(text_browser, 1)  # 1 = stretch factor
        
        # Add layouts to main layout
        layout.addLayout(header_layout)
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
