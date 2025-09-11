#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update checking utility for NeuralNetworkApp.
Provides functionality to check for updates and display them in an independent dialog.
"""

import os
import sys
import json
import requests
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import packaging.version

# Try to import PySide6 for GUI, fall back to console if not available
try:
    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                   QPushButton, QProgressBar, QTextEdit, 
                                   QDialogButtonBox, QMessageBox)
    from PySide6.QtCore import QThread, Signal, QTimer
    from PySide6.QtGui import QFont
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Add the src directory to the path to import version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from neuralnetworkapp.version import __version__


class UpdateChecker:
    """Handles update checking and notification."""
    
    def __init__(self):
        self.current_version = __version__
        self.github_api_url = "https://api.github.com/repos/Nsfr750/NeuralNetworkApp/releases/latest"
        self.github_repo_url = "https://github.com/Nsfr750/NeuralNetworkApp"
        self.cache_file = os.path.join(os.path.expanduser("~"), ".neuralnetworkapp_update_cache.json")
        self.cache_duration = timedelta(hours=24)  # Check for updates once per day
        
    def get_cached_update_info(self) -> Optional[Dict[str, Any]]:
        """Get cached update information if it's still valid."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    
                cached_time = datetime.fromisoformat(cached_data.get('cached_time', ''))
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_data
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        
        return None
    
    def cache_update_info(self, update_info: Dict[str, Any]) -> None:
        """Cache update information."""
        try:
            update_info['cached_time'] = datetime.now().isoformat()
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(update_info, f, indent=2)
        except OSError:
            pass
    
    def fetch_latest_release(self) -> Optional[Dict[str, Any]]:
        """Fetch the latest release information from GitHub."""
        try:
            response = requests.get(self.github_api_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError):
            return None
    
    def check_for_updates(self, force_check: bool = False) -> Optional[Dict[str, Any]]:
        """Check for updates, using cache if available."""
        if not force_check:
            cached_info = self.get_cached_update_info()
            if cached_info:
                return cached_info
        
        release_info = self.fetch_latest_release()
        if release_info:
            update_info = {
                'version': release_info.get('tag_name', '').lstrip('v'),
                'name': release_info.get('name', ''),
                'body': release_info.get('body', ''),
                'html_url': release_info.get('html_url', ''),
                'published_at': release_info.get('published_at', ''),
                'is_newer': False
            }
            
            # Check if the new version is actually newer
            try:
                latest_version = packaging.version.parse(update_info['version'])
                current_version = packaging.version.parse(self.current_version)
                update_info['is_newer'] = latest_version > current_version
            except (packaging.version.InvalidVersion, ValueError):
                # If version parsing fails, assume it's not newer
                update_info['is_newer'] = False
            
            self.cache_update_info(update_info)
            return update_info
        
        return None
    
    def is_update_available(self, force_check: bool = False) -> bool:
        """Check if an update is available."""
        update_info = self.check_for_updates(force_check)
        return update_info and update_info.get('is_newer', False)


class UpdateDialog:
    """Independent dialog for displaying update information."""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.update_checker = UpdateChecker()
        
    def show_update_dialog(self, force_check: bool = False) -> None:
        """Show the update dialog."""
        if GUI_AVAILABLE:
            self._show_gui_dialog(force_check)
        else:
            self._show_console_dialog(force_check)
    
    def _show_gui_dialog(self, force_check: bool = False) -> None:
        """Show GUI dialog for updates."""
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Update Checker - NeuralNetworkApp")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        # Main layout
        layout = QVBoxLayout(dialog)
        
        # Title
        title_label = QLabel("NeuralNetworkApp Update Checker")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Current version
        version_layout = QHBoxLayout()
        version_layout.addWidget(QLabel("Current Version:"))
        current_version_label = QLabel(self.update_checker.current_version)
        version_layout.addWidget(current_version_label)
        version_layout.addStretch()
        layout.addLayout(version_layout)
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Checking for updates...")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress)
        
        # Release notes
        layout.addWidget(QLabel("Release Notes:"))
        self.release_notes = QTextEdit()
        self.release_notes.setReadOnly(True)
        layout.addWidget(self.release_notes)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.download_button = QPushButton("Download Update")
        self.download_button.clicked.connect(lambda: self._open_download_page(dialog))
        self.download_button.setEnabled(False)
        
        check_again_button = QPushButton("Check Again")
        check_again_button.clicked.connect(lambda: self._refresh_updates())
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        
        button_layout.addWidget(self.download_button)
        button_layout.addWidget(check_again_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        # Check for updates in a separate thread
        def check_updates_thread():
            update_info = self.update_checker.check_for_updates(force_check)
            
            # Update GUI in main thread
            QTimer.singleShot(0, lambda: self._update_gui_with_results(update_info))
        
        threading.Thread(target=check_updates_thread, daemon=True).start()
        
        # Show the dialog
        dialog.exec_()
    
    def _update_gui_with_results(self, update_info):
        """Update the GUI with update check results."""
        self.progress.setRange(0, 100)  # Stop indeterminate progress
        self.progress.setValue(100)
        
        if not update_info:
            self.status_label.setText("Failed to check for updates")
            self.status_label.setStyleSheet("color: red;")
            self.release_notes.setText("Unable to connect to the update server.\nPlease check your internet connection and try again.")
            self.download_button.setEnabled(False)
            return
        
        latest_version = update_info.get('version', 'Unknown')
        is_newer = update_info.get('is_newer', False)
        
        if is_newer:
            self.status_label.setText(f"Update available: {latest_version}")
            self.status_label.setStyleSheet("color: green;")
            release_text = f"Version {latest_version} - {update_info.get('name', '')}\n"
            release_text += "="*50 + "\n\n"
            release_text += update_info.get('body', 'No release notes available.')
            self.release_notes.setText(release_text)
            self.download_button.setEnabled(True)
        else:
            self.status_label.setText(f"You're using the latest version: {latest_version}")
            self.status_label.setStyleSheet("color: blue;")
            release_text = f"You're already using the latest version ({latest_version}).\n\n"
            if update_info.get('body'):
                release_text += "Latest release notes:\n"
                release_text += "="*50 + "\n\n"
                release_text += update_info.get('body', '')
            self.release_notes.setText(release_text)
            self.download_button.setEnabled(False)
    
    def _refresh_updates(self):
        """Refresh update information."""
        self.status_label.setText("Checking for updates...")
        self.status_label.setStyleSheet("")
        self.release_notes.clear()
        self.progress.setRange(0, 0)  # Indeterminate progress
        self.download_button.setEnabled(False)
        
        def check_updates_thread():
            update_info = self.update_checker.check_for_updates(force_check=True)
            QTimer.singleShot(0, lambda: self._update_gui_with_results(update_info))
        
        threading.Thread(target=check_updates_thread, daemon=True).start()
    
    def _open_download_page(self, dialog):
        """Open the download page in the default browser."""
        import webbrowser
        webbrowser.open(self.update_checker.github_repo_url + "/releases/latest")
    
    def _show_console_dialog(self, force_check: bool = False) -> None:
        """Show console-based update information."""
        print("NeuralNetworkApp Update Checker")
        print("=" * 40)
        print(f"Current Version: {self.update_checker.current_version}")
        print("Checking for updates...")
        
        update_info = self.update_checker.check_for_updates(force_check)
        
        if not update_info:
            print("Failed to check for updates. Please check your internet connection.")
            return
        
        latest_version = update_info.get('version', 'Unknown')
        is_newer = update_info.get('is_newer', False)
        
        print(f"\nLatest Version: {latest_version}")
        
        if is_newer:
            print(f"✓ Update available! Version {latest_version} is ready to download.")
            print(f"\nRelease Notes:")
            print(update_info.get('body', 'No release notes available.'))
            print(f"\nDownload: {self.update_checker.github_repo_url}/releases/latest")
        else:
            print(f"✓ You're using the latest version.")
        
        input("\nPress Enter to continue...")


def check_for_updates(parent=None, force_check: bool = False) -> None:
    """
    Check for updates and show the update dialog.
    
    Args:
        parent: Parent window for the dialog (optional)
        force_check: Force check ignoring cache (default: False)
    """
    dialog = UpdateDialog(parent)
    dialog.show_update_dialog(force_check)


def is_update_available(force_check: bool = False) -> bool:
    """
    Check if an update is available without showing dialog.
    
    Args:
        force_check: Force check ignoring cache (default: False)
    
    Returns:
        bool: True if update is available, False otherwise
    """
    checker = UpdateChecker()
    return checker.is_update_available(force_check)


if __name__ == "__main__":
    # Test the update checker
    check_for_updates()
