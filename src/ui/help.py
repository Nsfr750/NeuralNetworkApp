from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, 
                             QTextBrowser, QDialogButtonBox)
from PySide6.QtGui import QDesktopServices
import webbrowser


def show_help(parent=None):
    """Show the help dialog.
    
    Args:
        parent: Parent widget for the dialog
    """
    dialog = HelpDialog(parent)
    dialog.exec()


class HelpDialog(QDialog):
    """Custom help dialog with documentation and links."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help - Neural Network Creator")
        self.setMinimumSize(700, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create text browser for the help content
        help_text = """
        <h2 style='text-align: center;'>Neural Network Creator - Help</h2>
        
        <h3>Getting Started</h3>
        <p>Welcome to Neural Network Creator! This application helps you create, train, and manage neural networks.</p>
        
        <h3>Documentation</h3>
        <p>For detailed documentation, please visit our <a href='https://github.com/Nsfr750/NeuralNetworkApp/wiki'>Wiki</a>.</p>
        
        <h3>Quick Links</h3>
        <ul>
            <li><a href='https://github.com/Nsfr750/NeuralNetworkApp/issues'>Report an Issue</a></li>
            <li><a href='https://github.com/Nsfr750/NeuralNetworkApp/discussions'>Community Discussions</a></li>
            <li><a href='https://github.com/Nsfr750/NeuralNetworkApp/releases'>Release Notes</a></li>
        </ul>
        
        <h3>Keyboard Shortcuts</h3>
        <ul>
            <li><b>Ctrl+N</b>: New Project</li>
            <li><b>Ctrl+O</b>: Open Project</li>
            <li><b>Ctrl+S</b>: Save Project</li>
            <li><b>F1</b>: Show Help</li>
        </ul>
        
        <h3>Need More Help?</h3>
        <p>If you need further assistance, please don't hesitate to open an issue on our <a href='https://github.com/Nsfr750/NeuralNetworkApp/issues'>GitHub repository</a>.</p>
        """
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(help_text)
        text_browser.setReadOnly(True)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        
        # Add widgets to layout
        layout.addWidget(text_browser)
        layout.addWidget(button_box)
        
        # Set layout
        self.setLayout(layout)
