"""
Neural Network Application Menu System

This module provides a command-line interface for the Neural Network application.
It allows users to interact with the application through a series of menus.
"""

import sys
from typing import List, Dict, Callable, Optional
import logging

class MenuItem:
    """Represents a single menu item with a title and action."""
    
    def __init__(self, title: str, action: Callable, requires_model: bool = False):
        """
        Initialize a menu item.
        
        Args:
            title: The display text for the menu item
            action: The function to call when this item is selected
            requires_model: Whether this action requires a loaded model
        """
        self.title = title
        self.action = action
        self.requires_model = requires_model


class Menu:
    """A hierarchical menu system for the Neural Network application."""
    
    def __init__(self, title: str, items: List[MenuItem] = None):
        """
        Initialize a menu with a title and optional items.
        
        Args:
            title: The title of the menu
            items: Optional list of menu items
        """
        self.title = title
        self.items = items or []
        self.parent = None
        self.logger = logging.getLogger(__name__)
    
    def add_item(self, title: str, action: Callable, requires_model: bool = False) -> 'MenuItem':
        """
        Add an item to the menu.
        
        Args:
            title: The display text for the menu item
            action: The function to call when this item is selected
            requires_model: Whether this action requires a loaded model
            
        Returns:
            The created MenuItem
        """
        item = MenuItem(title, action, requires_model)
        self.items.append(item)
        return item
    
    def add_submenu(self, title: str, submenu: 'Menu') -> 'Menu':
        """
        Add a submenu to this menu.
        
        Args:
            title: The display text for the submenu
            submenu: The Menu object to add as a submenu
            
        Returns:
            The submenu that was added
        """
        submenu.parent = self
        self.items.append(MenuItem(title, lambda: submenu.show()))
        return submenu
    
    def _print_header(self, title: str) -> None:
        """Print a formatted header."""
        width = 60
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width + "\n")
    
    def _print_menu_item(self, number: int, text: str, disabled: bool = False) -> None:
        """Print a formatted menu item."""
        prefix = "[!] " if disabled else ""
        print(f" {number:>2}. {prefix}{text}")
    
    def show(self) -> None:
        """Display the menu and handle user input with better visibility."""
        while True:
            self._clear_screen()
            self._print_header(self.title)
            
            # Display menu items
            for i, item in enumerate(self.items, 1):
                self._print_menu_item(i, item.title, item.requires_model and not self._is_model_loaded())
            
            # Add navigation options
            nav_text = "Back to previous menu" if self.parent else "Exit application"
            self._print_menu_item(len(self.items) + 1, nav_text)
            
            # Get user input
            try:
                choice = input("\n\033[1mEnter your choice (1-{}):\033[0m ".format(len(self.items) + 1)).strip()
                if not choice:
                    continue
                    
                choice_idx = int(choice) - 1
                
                # Handle navigation
                if choice_idx == len(self.items):
                    return  # Go back to parent menu
                
                # Validate choice
                if 0 <= choice_idx < len(self.items):
                    selected_item = self.items[choice_idx]
                    
                    # Check if model is required and loaded
                    if selected_item.requires_model and not self._is_model_loaded():
                        input("\n\033[91mError: No model is loaded. Please load a model first.\033[0m\n\nPress Enter to continue...")
                        continue
                    
                    # Execute the selected action
                    try:
                        self._clear_screen()
                        self._print_header(f"{self.title} > {selected_item.title}")
                        selected_item.action()
                        if selected_item.requires_model:
                            input("\n\033[92mOperation completed successfully!\033[0m\n\nPress Enter to continue...")
                    except Exception as e:
                        self.logger.exception("Error executing menu action")
                        input(f"\n\033[91mError: {str(e)}\033[0m\n\nPress Enter to continue...")
                else:
                    input("\n\033[93mInvalid choice. Please try again.\033[0m\n\nPress Enter to continue...")
                    
            except ValueError:
                input("\n\033[93mPlease enter a valid number.\033[0m\n\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\033[93mOperation cancelled by user.\033[0m")
                if input("Do you want to exit? (y/n): ").lower() == 'y':
                    sys.exit(0)
    
    def _clear_screen(self) -> None:
        """Clear the console screen in a cross-platform way."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        # This should be implemented to check your application's model state
        # For now, it's a placeholder that returns True
        return True


def create_main_menu() -> Menu:
    """
    Create and configure the main menu for the application.
    
    Returns:
        The configured main menu
    """
    # Create the main menu
    main_menu = Menu("Neural Network Application")
    
    # Data management submenu
    data_menu = Menu("Data Management")
    data_menu.add_item("Load Dataset", lambda: print("Loading dataset..."))
    data_menu.add_item("Preprocess Data", lambda: print("Preprocessing data..."), requires_model=True)
    data_menu.add_item("Augment Data", lambda: print("Augmenting data..."), requires_model=True)
    
    # Model submenu
    model_menu = Menu("Model Management")
    model_menu.add_item("Create New Model", lambda: print("Creating new model..."))
    model_menu.add_item("Load Model", lambda: print("Loading model..."))
    model_menu.add_item("Save Model", lambda: print("Saving model..."), requires_model=True)
    
    # Training submenu
    training_menu = Menu("Training")
    training_menu.add_item("Train Model", lambda: print("Training model..."), requires_model=True)
    training_menu.add_item("Evaluate Model", lambda: print("Evaluating model..."), requires_model=True)
    training_menu.add_item("View Training History", lambda: print("Showing training history..."), requires_model=True)
    
    # Prediction submenu
    prediction_menu = Menu("Prediction")
    prediction_menu.add_item("Single Prediction", lambda: print("Making single prediction..."), requires_model=True)
    prediction_menu.add_item("Batch Prediction", lambda: print("Making batch predictions..."), requires_model=True)
    
    # Add all submenus to main menu
    main_menu.add_submenu("Data Management", data_menu)
    main_menu.add_submenu("Model Management", model_menu)
    main_menu.add_submenu("Training", training_menu)
    main_menu.add_submenu("Prediction", prediction_menu)
    
    # Add direct actions
    main_menu.add_item("View Model Summary", lambda: print("Displaying model summary..."), requires_model=True)
    main_menu.add_item("Export Model", lambda: print("Exporting model..."), requires_model=True)
    
    return main_menu


def run() -> None:
    """Initialize and run the menu system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and show the main menu
    menu = create_main_menu()
    
    try:
        menu.show()
    except KeyboardInterrupt:
        print("\nExiting application...")
        sys.exit(0)


if __name__ == "__main__":
    run()
