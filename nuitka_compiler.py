#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuitka compilation script for Neural Network App - Safe Mode
© Copyright 2025 Nsfr750 - All rights reserved

This script compiles the Neural Network App using Nuitka with safer options.
"""

import os
import sys
import subprocess
import platform
import logging
import time
from datetime import datetime
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.absolute()

def get_python_executable():
    """Get the Python executable path."""
    return sys.executable

def get_output_dir():
    """Get the output directory for compiled executables."""
    return get_project_root() / "dist"

def get_data_dirs():
    """Get data directories that need to be included."""
    project_root = get_project_root()
    return [
        project_root / "assets",
        project_root / "cli",
        project_root / "docs",
        project_root / "data",
        project_root / "checkpoints",
        project_root / "logs"
    ]

def get_icon_path():
    """Get the application icon path."""
    icon_path = get_project_root() / "assets" / "icon.ico"
    return icon_path if icon_path.exists() else None

def get_app_config():
    """
    Get application configuration.
    
    Returns:
        dict: Application configuration
    """
    return {
        "name": "Neural Network",
        "version": "0.1.0",
        "company": "Tuxxle",
        "description": "Neural Network Creator",
        "copyright": "© Copyright 2025 Nsfr750 - All rights reserved",
        "output_filename": "NeuralNetwork"
    }

def get_nuitka_command(compilation_mode="standalone", debug=False):
    """
    Build the Nuitka compilation command with enhanced options.
    
    Args:
        compilation_mode (str): "standalone" or "onefile"
        debug (bool): Whether to include debug information
    
    Returns:
        list: Nuitka command as a list of arguments
    """
    project_root = get_project_root()
    main_script = project_root / "main.py"
    output_dir = get_output_dir()
    icon_path = get_icon_path()
    config = get_app_config()
    
    if not main_script.exists():
        raise FileNotFoundError(f"Main script not found: {main_script}")
    
    # Base command with enhanced options
    cmd = [
        get_python_executable(),
        "-m",
        "nuitka",
        "--standalone" if compilation_mode == "standalone" else "--onefile",
        "--follow-imports",
        "--follow-import-to=src",
        "--follow-import-to=src.neuralnetworkapp",
        "--include-package=src",
        "--include-package=src.neuralnetworkapp",
        "--include-package=src.network_builder",
        "--include-package=src.ui",
        "--include-package=src.utils",
        "--windows-console-mode=disable",
        f"--output-dir={output_dir}",
        f"--output-filename={config['output_filename']}",
        f"--company-name={config['company']}",
        f"--product-name={config['name']}",
        f"--file-version={config['version']}.0",
        f"--product-version={config['version']}.0",
        f"--file-description={config['description']}",
        f"--copyright={config['copyright']}",
        "--remove-output",
        "--jobs=8",
        "--lto=yes",
        "--plugin-enable=numpy",
        "--plugin-enable=pyside6",
        "--plugin-enable=torch",
        "--assume-yes-for-downloads",
        "--no-pyi-file"
    ]
    
    # Add icon if available
    if icon_path:
        cmd.append(f"--windows-icon-from-ico={icon_path}")
    
    # Add data directories
    for data_dir in get_data_dirs():
        if data_dir.exists():
            cmd.append(f"--include-data-dir={data_dir}={data_dir.name}")
    
    # Add additional data files
    additional_files = [
        project_root / "requirements.txt",
        project_root / "README.md",
        project_root / "LICENSE",
        project_root / "CHANGELOG.md"
    ]
    
    for file_path in additional_files:
        if file_path.exists():
            cmd.append(f"--include-data-files={file_path}={file_path.name}")
    
    # Add debug flags if needed
    if debug:
        cmd.extend([
            "--debug",
            "--unstripped",
            "--profile"
        ])
    else:
        # Optimization flags for release builds
        cmd.extend([
            "--follow-stdlib",
            "--python-flag=no_site",
            "--python-flag=no_warnings"
        ])
    
    # Add platform-specific options
    if platform.system() == "Windows":
        cmd.extend([
            "--disable-console" if compilation_mode == "standalone" else ""
        ])
        cmd = [x for x in cmd if x]  # Remove empty strings
    
    # Add the main script
    cmd.append(str(main_script))
    
    return cmd

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Check for Nuitka
    try:
        import nuitka
        print("[OK] Nuitka found")
    except ImportError:
        print("[ERROR] Nuitka not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka"])
    
    # Check for PySide6
    try:
        import PySide6
        print("[OK] PySide6 found")
    except ImportError:
        print("[ERROR] PySide6 not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6"])
    
    # Check for torch
    try:
        import torch
        print("[OK] PyTorch found")
    except ImportError:
        print("[ERROR] PyTorch not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    
    # Check for numpy
    try:
        import numpy
        print("[OK] NumPy found")
    except ImportError:
        print("[ERROR] NumPy not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
    # Check for matplotlib
    try:
        import matplotlib
        print("[OK] Matplotlib found")
    except ImportError:
        print("[ERROR] Matplotlib not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    
    print("[OK] All dependencies checked")
    return True


def setup_logging():
    """
    Setup logging configuration.
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging
    log_file = logs_dir / f"nuitka_compilation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def compile_project(mode="standalone", debug=False):
    """
    Compile the project using Nuitka with enhanced error handling and logging.
    
    Args:
        mode (str): Compilation mode ("standalone" or "onefile")
        debug (bool): Whether to include debug information
    
    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    logger = setup_logging()
    config = get_app_config()
    start_time = time.time()
    
    logger.info(f"Starting {config['name']} compilation (Safe Mode)...")
    logger.info(f"Version: {config['version']}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Debug: {debug}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info("-" * 50)
    
    try:
        # Check and install dependencies
        logger.info("Checking dependencies...")
        if not check_dependencies():
            logger.error("Dependencies check failed. Aborting compilation.")
            return False
        
        # Create output directory
        output_dir = get_output_dir()
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Get compilation command
        cmd = get_nuitka_command(mode, debug)
        logger.info("Compilation command:")
        logger.info(" ".join(cmd))
        logger.info("-" * 50)
        
        # Run compilation
        logger.info("Starting compilation (this may take a while)...")
        
        try:
            result = subprocess.run(cmd, check=False, cwd=get_project_root(), 
                                  capture_output=True, text=True)
            
            # Log output regardless of success/failure
            if result.stdout:
                logger.info("Compilation output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            if result.stderr:
                logger.error("Compilation errors:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            
            if result.returncode == 0:
                elapsed_time = time.time() - start_time
                logger.info("✓ Compilation completed successfully!")
                logger.info(f"Total time: {elapsed_time:.2f} seconds")
                logger.info(f"Output directory: {output_dir}")
                
                # List the generated files
                if output_dir.exists():
                    logger.info("Generated files:")
                    for item in output_dir.iterdir():
                        if item.is_file():
                            size = item.stat().st_size
                            logger.info(f"  - {item.name} ({size:,} bytes)")
                        elif item.is_dir():
                            logger.info(f"  - {item.name}/")
                
                logger.info("-" * 50)
                logger.info("Compilation completed successfully!")
                return True
            else:
                logger.error(f"[ERROR] Compilation failed with return code: {result.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Compilation failed with error code: {e.returncode}")
            if e.stderr:
                logger.error("Error output:")
                logger.error(e.stderr)
            if e.stdout:
                logger.info("Standard output:")
                logger.info(e.stdout)
            return False
        except KeyboardInterrupt:
            logger.warning("[WARNING] Compilation interrupted by user.")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error during compilation: {e}")
            logger.exception("Full traceback:")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Critical error during compilation setup: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compile Neural Network App with Nuitka (Safe Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nuitka_compiler.py                    # Compile standalone (default)
  python nuitka_compiler.py --mode onefile     # Compile as single executable
  python nuitka_compiler.py --debug            # Compile with debug information
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["standalone", "onefile"],
        default="standalone",
        help="Compilation mode (default: standalone)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include debug information"
    )
    
    args = parser.parse_args()
    
    # Compile the project
    success = compile_project(args.mode, args.debug)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
