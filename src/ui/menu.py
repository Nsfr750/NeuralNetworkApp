from PySide6.QtCore import Qt, QProcess, Signal
from PySide6.QtWidgets import QMenuBar, QMenu, QMessageBox
from PySide6.QtGui import QAction, QActionGroup
import webbrowser
import os
import sys

# Local imports
from .about import show_about
from .sponsor import show_sponsor
from .help import show_help
from .lang_mgr import get_language_manager, get_text, set_language

class AppMenuBar(QMenuBar):
    """Custom menu bar for the Neural Network Creator application."""
    
    # Signal emitted when language is changed
    language_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.lang_manager = get_language_manager()
        self.setup_menus()
    
    def setup_menus(self):
        """Set up the application menus."""
        # File Menu
        file_menu = self.addMenu(get_text('menu.file', '&File'))
        
        # New Action
        new_action = file_menu.addAction(get_text('menu.new', 'New'))
        new_action.setShortcut('Ctrl+N')
        
        # Open Action
        open_action = file_menu.addAction(get_text('menu.open', 'Open...'))
        open_action.setShortcut('Ctrl+O')
        
        # Save Action
        save_action = file_menu.addAction(get_text('menu.save', 'Save'))
        save_action.setShortcut('Ctrl+S')
        
        # Save As Action
        save_as_action = file_menu.addAction(get_text('menu.save_as', 'Save As...'))
        save_as_action.setShortcut('Ctrl+Shift+S')
        
        file_menu.addSeparator()
        
        # Exit Action
        exit_action = file_menu.addAction(get_text('menu.exit', 'Exit'))
        exit_action.setShortcut('Alt+F4')
        exit_action.triggered.connect(self.close_application)
        
        # Edit Menu
        edit_menu = self.addMenu(get_text('menu.edit', '&Edit'))
        
        # Undo Action
        undo_action = edit_menu.addAction(get_text('menu.undo', 'Undo'))
        undo_action.setShortcut('Ctrl+Z')
        
        # Redo Action
        redo_action = edit_menu.addAction(get_text('menu.redo', 'Redo'))
        redo_action.setShortcut('Ctrl+Y')
        
        edit_menu.addSeparator()
        
        # Cut Action
        cut_action = edit_menu.addAction(get_text('menu.cut', 'Cut'))
        cut_action.setShortcut('Ctrl+X')
        
        # Copy Action
        copy_action = edit_menu.addAction(get_text('menu.copy', 'Copy'))
        copy_action.setShortcut('Ctrl+C')
        
        # Paste Action
        paste_action = edit_menu.addAction(get_text('menu.paste', 'Paste'))
        paste_action.setShortcut('Ctrl+V')
        
        # Delete Action
        delete_action = edit_menu.addAction(get_text('menu.delete', 'Delete'))
        delete_action.setShortcut('Del')
        
        edit_menu.addSeparator()
        
        # Select All Action
        select_all_action = edit_menu.addAction(get_text('menu.select_all', 'Select All'))
        select_all_action.setShortcut('Ctrl+A')
        
        # View Menu
        view_menu = self.addMenu(get_text('menu.view', '&View'))
        
        # Language Submenu
        language_menu = view_menu.addMenu(get_text('menu.language', '&Language'))
        
        # Language Actions
        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        
        # English
        en_action = QAction(get_text('menu.english', 'English'), self)
        en_action.setCheckable(True)
        en_action.setChecked(True)  # Default
        en_action.triggered.connect(lambda: self.change_language('en'))
        language_menu.addAction(en_action)
        self.language_group.addAction(en_action)
        
        # Italian
        it_action = QAction(get_text('menu.italian', 'Italiano'), self)
        it_action.setCheckable(True)
        it_action.triggered.connect(lambda: self.change_language('it'))
        language_menu.addAction(it_action)
        self.language_group.addAction(it_action)
        
        view_menu.addSeparator()
        
        # Theme Submenu
        theme_menu = view_menu.addMenu(get_text('menu.theme', '&Theme'))
        
        # Theme Actions
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        
        # System Theme
        system_theme = QAction(get_text('menu.system', 'System'), self)
        system_theme.setCheckable(True)
        system_theme.setChecked(True)
        theme_menu.addAction(system_theme)
        theme_group.addAction(system_theme)
        
        # Dark Theme
        dark_theme = QAction(get_text('menu.dark', 'Dark'), self)
        dark_theme.setCheckable(True)
        theme_menu.addAction(dark_theme)
        theme_group.addAction(dark_theme)
        
        # Tools Menu
        tools_menu = self.addMenu(get_text('menu.tools', '&Tools'))
        
        # View Logs Action
        view_logs_action = tools_menu.addAction(get_text('menu.view_logs', 'View Logs'))
        view_logs_action.triggered.connect(self.view_logs)

        tools_menu.addSeparator()
        
        # Help Menu
        help_menu = self.addMenu(get_text('menu.help', '&Help'))
        
        # Help Action
        help_action = help_menu.addAction(get_text('menu.help', 'Help'))
        help_action.setShortcut('F1')
        help_action.triggered.connect(self.show_help)
        
        # Wiki Action
        wiki_action = help_menu.addAction(get_text('menu.documentation', 'Documentation (Wiki)'))
        wiki_action.triggered.connect(self.open_wiki)
        
        # Add separator
        help_menu.addSeparator()
        
        # About Action
        about_action = help_menu.addAction(get_text('menu.about', 'About'))
        about_action.triggered.connect(self.show_about)
        
        # Sponsor Action
        sponsor_action = help_menu.addAction(get_text('menu.sponsor', 'Sponsor'))
        sponsor_action.triggered.connect(self.show_sponsor)
    
    def change_language(self, lang_code):
        """Change the application language."""
        try:
            # Set the new language
            set_language(lang_code)
            
            # Update the UI
            self.retranslate_ui()
            
            # Emit signal to notify parent of language change
            self.language_changed.emit()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                get_text('app.error', 'Error'),
                get_text('app.language_change_error', 'Failed to change language: {error}').format(error=str(e))
            )
    
    def retranslate_ui(self):
        """Retranslate the UI elements."""
        # File menu
        file_menu = self.actions()[0].menu()
        file_menu.setTitle(get_text('menu.file', '&File'))
        
        # Edit menu
        edit_menu = self.actions()[1].menu()
        edit_menu.setTitle(get_text('menu.edit', '&Edit'))
        
        # View menu
        view_menu = self.actions()[2].menu()
        view_menu.setTitle(get_text('menu.view', '&View'))
        
        # Tools menu
        tools_menu = self.actions()[3].menu()
        tools_menu.setTitle(get_text('menu.tools', '&Tools'))
        
        # Help menu
        help_menu = self.actions()[4].menu()
        help_menu.setTitle(get_text('menu.help', '&Help'))
    
    def close_application(self):
        """Close the application."""
        reply = QMessageBox.question(
            self,
            get_text('app.confirm_exit', 'Confirm Exit'),
            get_text('app.confirm_exit_text', 'Are you sure you want to exit?'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.parent.close()
    
    def show_about(self):
        """Show the about dialog."""
        show_about(self.parent)
    
    def show_sponsor(self):
        """Show the sponsor dialog."""
        show_sponsor(self.parent)
    
    def show_help(self):
        """Show the help dialog."""
        show_help(self.parent)
    
    def open_wiki(self):
        """Open the project wiki in the default web browser."""
        webbrowser.open("https://github.com/Nsfr750/NeuralNetworkApp/wiki")
    
    def view_logs(self):
        """Open the log viewer dialog."""
        try:
            from .view_log import LogViewer
            log_viewer = LogViewer(self.parent)
            log_viewer.language_changed.connect(self.retranslate_ui)
            log_viewer.exec_()
        except Exception as e:
            # Fallback to opening in a separate process if in-process fails
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_script = os.path.join(current_dir, 'view_log.py')
            
            if os.path.exists(log_script):
                python = sys.executable
                QProcess.startDetached(python, [log_script])
            else:
                QMessageBox.critical(
                    self,
                    get_text('app.error', 'Error'),
                    get_text('app.log_viewer_error', 'Failed to open log viewer')
                )
                QMessageBox.warning(
                    self.parent,
                    "Log Viewer Not Found",
                    f"Could not find log viewer at: {log_script}"
                )
        except Exception as e:
            QMessageBox.critical(
                self.parent,
                "Error",
                f"Failed to open log viewer: {str(e)}"
            )
