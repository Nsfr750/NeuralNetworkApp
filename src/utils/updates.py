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

# Try to import tkinter for GUI, fall back to console if not available
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
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
        dialog = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        dialog.title("Update Checker - NeuralNetworkApp")
        dialog.geometry("600x500")
        dialog.resizable(True, True)
        
        # Center the dialog
        dialog.transient(self.parent) if self.parent else None
        dialog.grab_set() if self.parent else None
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="NeuralNetworkApp Update Checker", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Current version
        ttk.Label(main_frame, text="Current Version:").grid(row=1, column=0, sticky=tk.W, pady=2)
        current_version_label = ttk.Label(main_frame, text=self.update_checker.current_version)
        current_version_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Status
        ttk.Label(main_frame, text="Status:").grid(row=2, column=0, sticky=tk.W, pady=2)
        status_label = ttk.Label(main_frame, text="Checking for updates...")
        status_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Progress bar
        progress = ttk.Progressbar(main_frame, mode='indeterminate')
        progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        progress.start()
        
        # Update info frame (initially hidden)
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(4, weight=1)
        
        # Release notes text area
        ttk.Label(info_frame, text="Release Notes:").grid(row=0, column=0, sticky=tk.W, pady=2)
        release_notes = scrolledtext.ScrolledText(info_frame, height=10, width=60)
        release_notes.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(1, weight=1)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        download_button = ttk.Button(buttons_frame, text="Download Update", 
                                   command=lambda: self._open_download_page(dialog))
        download_button.pack(side=tk.LEFT, padx=5)
        
        check_again_button = ttk.Button(buttons_frame, text="Check Again", 
                                      command=lambda: self._refresh_updates(dialog, status_label, 
                                                                         progress, info_frame, release_notes))
        check_again_button.pack(side=tk.LEFT, padx=5)
        
        close_button = ttk.Button(buttons_frame, text="Close", 
                                command=dialog.destroy)
        close_button.pack(side=tk.LEFT, padx=5)
        
        # Check for updates in a separate thread
        def check_updates_thread():
            update_info = self.update_checker.check_for_updates(force_check)
            
            # Update GUI in main thread
            dialog.after(0, lambda: self._update_gui_with_results(dialog, status_label, progress, 
                                                               info_frame, release_notes, download_button, 
                                                               update_info))
        
        threading.Thread(target=check_updates_thread, daemon=True).start()
        
        # Run the dialog
        dialog.mainloop()
    
    def _update_gui_with_results(self, dialog, status_label, progress, info_frame, 
                               release_notes, download_button, update_info):
        """Update the GUI with update check results."""
        progress.stop()
        progress.grid_remove()
        
        if not update_info:
            status_label.config(text="Failed to check for updates", foreground="red")
            release_notes.insert(tk.END, "Unable to connect to the update server.\nPlease check your internet connection and try again.")
            download_button.config(state='disabled')
            return
        
        latest_version = update_info.get('version', 'Unknown')
        is_newer = update_info.get('is_newer', False)
        
        if is_newer:
            status_label.config(text=f"Update available: {latest_version}", foreground="green")
            release_notes.insert(tk.END, f"Version {latest_version} - {update_info.get('name', '')}\n")
            release_notes.insert(tk.END, "="*50 + "\n\n")
            release_notes.insert(tk.END, update_info.get('body', 'No release notes available.'))
            download_button.config(state='normal')
        else:
            status_label.config(text=f"You're using the latest version: {latest_version}", foreground="blue")
            release_notes.insert(tk.END, f"You're already using the latest version ({latest_version}).\n\n")
            if update_info.get('body'):
                release_notes.insert(tk.END, "Latest release notes:\n")
                release_notes.insert(tk.END, "="*50 + "\n\n")
                release_notes.insert(tk.END, update_info.get('body', ''))
            download_button.config(state='disabled')
        
        release_notes.config(state='disabled')
    
    def _refresh_updates(self, dialog, status_label, progress, info_frame, release_notes):
        """Refresh update information."""
        status_label.config(text="Checking for updates...")
        release_notes.config(state='normal')
        release_notes.delete(1.0, tk.END)
        progress.grid()
        progress.start()
        
        def check_updates_thread():
            update_info = self.update_checker.check_for_updates(force_check=True)
            dialog.after(0, lambda: self._update_gui_with_results(dialog, status_label, progress, 
                                                               info_frame, release_notes, None, update_info))
        
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
