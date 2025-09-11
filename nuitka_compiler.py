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
    # Only include directories that typically contain data files (not Python source directories)
    data_dirs = [
        project_root / "assets",
        project_root / "docs",
        project_root / "logs",
    ]
    
    # Only include directories that actually contain files
    valid_dirs = []
    for data_dir in data_dirs:
        if data_dir.exists():
            # Check if directory contains any files
            has_files = any(data_dir.iterdir())
            if has_files:
                valid_dirs.append(data_dir)
    
    return valid_dirs

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
    ]
    
    # Add follow-imports options only for standalone mode
    if compilation_mode == "standalone":
        cmd.extend([
            "--follow-imports",
            "--follow-import-to=src",
            "--follow-import-to=src.neuralnetworkapp",
        ])
    
    # Add base options
    cmd.extend([
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
        "--plugin-enable=pyside6",
        "--assume-yes-for-downloads",
        "--no-pyi-file"
    ])
    
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
    Compile the project using Nuitka with enhanced error handling and real-time progress.
    
    Args:
        mode (str): Compilation mode ("standalone" or "onefile")
        debug (bool): Whether to include debug information
    
    Returns:
        bool: True if compilation succeeded, False otherwise
    """
    logger = setup_logging()
    config = get_app_config()
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {config['name']} compilation (Safe Mode)...")
    print(f"📋 Version: {config['version']}")
    print(f"🔧 Mode: {mode}")
    print(f"🐛 Debug: {debug}")
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"{'='*60}\n")
    
    logger.info(f"Starting {config['name']} compilation (Safe Mode)...")
    logger.info(f"Version: {config['version']}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Debug: {debug}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info("-" * 50)
    
    try:
        # Check and install dependencies
        print("🔍 Checking dependencies...")
        logger.info("Checking dependencies...")
        if not check_dependencies():
            print("❌ Dependencies check failed. Aborting compilation.")
            logger.error("Dependencies check failed. Aborting compilation.")
            return False
        
        # Create output directory
        output_dir = get_output_dir()
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Get compilation command
        cmd = get_nuitka_command(mode, debug)
        print(f"\n🔨 Compilation command prepared")
        logger.info("Compilation command:")
        logger.info(" ".join(cmd))
        logger.info("-" * 50)
        
        # Run compilation with real-time progress
        print("\n⏳ Starting compilation (this may take a while)...")
        print("📊 Progress will be shown below:\n")
        logger.info("Starting compilation (this may take a while)...")
        
        try:
            # Run subprocess with real-time output
            process = subprocess.Popen(
                cmd, 
                cwd=get_project_root(),
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Track compilation stages
            compilation_stages = {
                'parsing': False,
                'compiling': False,
                'linking': False,
                'creating_binary': False
            }
            
            line_count = 0
            start_progress_time = time.time()
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    line_count += 1
                    
                    # Track compilation stages
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in ['parsing', 'parse']):
                        if not compilation_stages['parsing']:
                            print("📖 Parsing source code...")
                            compilation_stages['parsing'] = True
                    
                    elif any(keyword in line_lower for keyword in ['compiling', 'compile']):
                        if not compilation_stages['compiling']:
                            print("⚙️  Compiling modules...")
                            compilation_stages['compiling'] = True
                    
                    elif any(keyword in line_lower for keyword in ['linking', 'link']):
                        if not compilation_stages['linking']:
                            print("🔗 Linking objects...")
                            compilation_stages['linking'] = True
                    
                    elif any(keyword in line_lower for keyword in ['creating', 'binary', 'executable']):
                        if not compilation_stages['creating_binary']:
                            print("🎯 Creating binary executable...")
                            compilation_stages['creating_binary'] = True
                    
                    # Show progress every 50 lines or every 30 seconds
                    if line_count % 50 == 0:
                        elapsed = time.time() - start_progress_time
                        print(f"📈 Progress: {line_count} lines processed ({elapsed:.1f}s elapsed)")
                    
                    # Print important lines to terminal
                    if any(keyword in line_lower for keyword in ['error', 'warning', 'failed', 'success', 'completed']):
                        if 'error' in line_lower or 'failed' in line_lower:
                            print(f"❌ {line}")
                        elif 'success' in line_lower or 'completed' in line_lower:
                            print(f"✅ {line}")
                        else:
                            print(f"⚠️  {line}")
                    
                    # Log all lines
                    logger.info(f"  {line}")
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                elapsed_time = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"✅ Compilation completed successfully!")
                print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
                print(f"📁 Output directory: {output_dir}")
                print(f"📊 Lines processed: {line_count}")
                
                # List the generated files
                if output_dir.exists():
                    print(f"\n📦 Generated files:")
                    for item in output_dir.iterdir():
                        if item.is_file():
                            size = item.stat().st_size
                            size_mb = size / (1024 * 1024)
                            print(f"  📄 {item.name} ({size:,.0f} bytes / {size_mb:.2f} MB)")
                        elif item.is_dir():
                            print(f"  📁 {item.name}/")
                
                print(f"{'='*60}")
                
                logger.info("✓ Compilation completed successfully!")
                logger.info(f"Total time: {elapsed_time:.2f} seconds")
                logger.info(f"Output directory: {output_dir}")
                
                return True
            else:
                print(f"\n❌ Compilation failed with return code: {return_code}")
                logger.error(f"[ERROR] Compilation failed with return code: {return_code}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Compilation failed with error code: {e.returncode}")
            logger.error(f"[ERROR] Compilation failed with error code: {e.returncode}")
            if e.stderr:
                print("Error output:")
                print(e.stderr)
                logger.error("Error output:")
                logger.error(e.stderr)
            if e.stdout:
                print("Standard output:")
                print(e.stdout)
                logger.info("Standard output:")
                logger.info(e.stdout)
            return False
        except KeyboardInterrupt:
            print("\n⚠️  Compilation interrupted by user.")
            logger.warning("[WARNING] Compilation interrupted by user.")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during compilation: {e}")
            logger.error(f"[ERROR] Unexpected error during compilation: {e}")
            logger.exception("Full traceback:")
            return False
            
    except Exception as e:
        print(f"❌ Critical error during compilation setup: {e}")
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
