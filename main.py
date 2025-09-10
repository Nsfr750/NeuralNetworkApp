#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Creator - A PySide6 application for creating and managing neural networks
© Copyright 2025 Nsfr750 - All rights reserved
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Initialize logger
from src.ui.logger import get_logger
logger = get_logger(__name__)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

# PySide6 imports
from PySide6.QtCore import Qt, QSize, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QIcon, QFont, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QFileDialog, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QProgressBar, QCheckBox, QLineEdit,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QSpacerItem, QFrame, QDialog, QDialogButtonBox
)

# UI Components
from src.ui.menu import AppMenuBar
from src.ui.lang_mgr import get_language_manager, get_text

# Matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Custom modules
from src.neuralnetworkapp.models import NeuralNetwork, create_model
from src.neuralnetworkapp.training import Trainer
from src.neuralnetworkapp.data import load_tabular_data, create_data_loaders, TabularDataset
from src.neuralnetworkapp.utils import (
    save_model, load_model, plot_training_history,
    save_config, load_config, count_parameters, set_seed
)

# Set random seed for reproducibility
set_seed(42)

# Model is now imported from model.py

class TrainingThread(QThread):
    """Thread for running model training in the background."""
    progress_updated = Signal(int, float, float)  # epoch, train_loss, val_loss
    log_message = Signal(str)
    training_finished = Signal(dict)  # training history
    training_error = Signal(str)  # error message
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        weight_decay: float = 0.0,
        loss_fn: str = 'cross_entropy',
        metrics: List[str] = None,
        device: str = None
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.metrics = metrics or ['accuracy']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self._stop_requested = False
        self.current_epoch = 0
        
    def stop(self):
        """Request the training to stop after the current epoch."""
        self._stop_requested = True
        
    def run(self):
        """Run the training loop."""
        try:
            logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
            logger.debug(f"Training configuration: optimizer={self.optimizer_name}, "
                       f"learning_rate={self.learning_rate}, weight_decay={self.weight_decay}, "
                       f"loss_fn={self.loss_fn}, metrics={self.metrics}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            logger.debug(f"Moved model to {self.device}")
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                device=self.device,
                optimizer=self.optimizer_name,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                loss_fn=self.loss_fn,
                metrics=self.metrics
            )
            
            # Custom callback for progress updates
            history = {'train_loss': [], 'val_loss': []}
            
            for epoch in range(self.epochs):
                if self._stop_requested:
                    logger.info("Training stopped by user")
                    self.log_message.emit("Training stopped by user.")
                    return
                    
                self.current_epoch = epoch
                logger.debug(f"Starting epoch {epoch+1}/{self.epochs}")
                
                # Train for one epoch
                train_loss, train_metrics = trainer.train_epoch(self.train_loader)
                history['train_loss'].append(train_loss)
                
                # Evaluate on validation set if available
                val_loss = None
                if self.val_loader is not None:
                    val_loss, val_metrics = trainer.evaluate(self.val_loader)
                    history['val_loss'].append(val_loss)
                
                # Emit progress update
                self.progress_updated.emit(
                    epoch + 1,
                    train_loss,
                    val_loss if val_loss is not None else 0.0
                )
                
                # Log metrics
                log_msg = f"Epoch {epoch+1}/{self.epochs} - "
                log_msg += f"train_loss: {train_loss:.4f}"
                
                if val_loss is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                
                logger.info(log_msg)
                self.log_message.emit(log_msg)
            
            # Training completed
            logger.info(f"Training completed successfully after {self.epochs} epochs")
            self.training_finished.emit(history)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.training_error.emit(error_msg)
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Request the training to stop after the current epoch."""
        self._stop_requested = True

class NeuralNetworkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing NeuralNetworkApp")
        self.model = None
        self.training_thread = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Initialize language manager
        self.lang_manager = get_language_manager()
        
        try:
            self.init_ui()
            self.setup_connections()
            logger.info("UI and connections initialized successfully")
        except Exception as e:
            logger.critical("Failed to initialize application", exc_info=True)
            raise
        
    def init_ui(self):
        # Main window setup
        self.setWindowTitle('Neural Network Creator')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create and set menu bar
        self.setMenuBar(AppMenuBar(self))
        
        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Model tab
        self.model_tab = QWidget()
        self.tabs.addTab(self.model_tab, 'Model')
        self.setup_model_tab()
        
        # Training tab
        self.training_tab = QWidget()
        self.tabs.addTab(self.training_tab, 'Training')
        self.setup_training_tab()
        
        # Data tab
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, 'Data')
        self.setup_data_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Ready')
        
    def setup_model_tab(self):
        # Model architecture controls
        layout = QVBoxLayout(self.model_tab)
        
        # Model configuration group
        model_group = QGroupBox('Model Configuration')
        model_layout = QFormLayout()
        
        # Input size
        self.input_size = QSpinBox()
        self.input_size.setRange(1, 10000)
        self.input_size.setValue(784)  # Default for MNIST
        model_layout.addRow('Input Size:', self.input_size)
        
        # Hidden layers
        self.hidden_layers = QLineEdit('128, 64')
        model_layout.addRow('Hidden Layers (comma-separated):', self.hidden_layers)
        
        # Output size
        self.output_size = QSpinBox()
        self.output_size.setRange(1, 1000)
        self.output_size.setValue(10)  # Default for MNIST
        model_layout.addRow('Output Size:', self.output_size)
        
        # Activation function
        self.activation = QComboBox()
        self.activation.addItems(['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'])
        model_layout.addRow('Activation:', self.activation)
        
        # Dropout
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 1.0)
        self.dropout.setSingleStep(0.1)
        self.dropout.setValue(0.0)
        model_layout.addRow('Dropout (0 to disable):', self.dropout)
        
        # Batch normalization
        self.batch_norm = QCheckBox()
        model_layout.addRow('Batch Normalization:', self.batch_norm)
        
        model_group.setLayout(model_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.create_btn = QPushButton('Create Model')
        self.visualize_btn = QPushButton('Visualize Model')
        button_layout.addWidget(self.create_btn)
        button_layout.addWidget(self.visualize_btn)
        
        # Model summary
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        
        # Add to layout
        layout.addWidget(model_group)
        layout.addLayout(button_layout)
        layout.addWidget(QLabel('Model Summary:'))
        layout.addWidget(self.model_summary)
        
    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        
        # Training controls
        train_group = QGroupBox('Training Configuration')
        train_layout = QFormLayout()
        
        # Optimizer
        self.optimizer = QComboBox()
        self.optimizer.addItems(['adam', 'sgd', 'rmsprop'])
        train_layout.addRow('Optimizer:', self.optimizer)
        
        # Learning rate
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 1.0)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setDecimals(4)
        train_layout.addRow('Learning Rate:', self.learning_rate)
        
        # Weight decay (L2 regularization)
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 0.1)
        self.weight_decay.setSingleStep(0.0001)
        self.weight_decay.setValue(0.0)
        self.weight_decay.setDecimals(5)
        train_layout.addRow('Weight Decay (L2):', self.weight_decay)
        
        # Epochs
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(10)
        train_layout.addRow('Epochs:', self.epochs)
        
        # Batch size
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(32)
        train_layout.addRow('Batch Size:', self.batch_size)
        
        # Loss function
        self.loss_fn = QComboBox()
        self.loss_fn.addItems(['cross_entropy', 'mse', 'l1', 'bce'])
        train_layout.addRow('Loss Function:', self.loss_fn)
        
        # Device selection
        self.device = QComboBox()
        self.device.addItems(['auto', 'cpu', 'cuda'])
        train_layout.addRow('Device:', self.device)
        
        train_group.setLayout(train_layout)
        
        # Training controls
        control_layout = QHBoxLayout()
        self.train_btn = QPushButton('Start Training')
        self.stop_btn = QPushButton('Stop Training')
        self.save_btn = QPushButton('Save Model')
        self.load_btn = QPushButton('Load Model')
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.load_btn)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        
        # Training log
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        
        # Add to layout
        layout.addWidget(train_group)
        layout.addLayout(control_layout)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel('Training Log:'))
        layout.addWidget(self.training_log)
        
    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)
        
        # Data loading
        data_group = QGroupBox('Data Loading')
        data_layout = QVBoxLayout()
        
        # Dataset selection
        dataset_layout = QHBoxLayout()
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText('Path to dataset (CSV or directory)')
        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(self.dataset_path)
        dataset_layout.addWidget(browse_btn)
        
        # Target column (for CSV)
        self.target_column = QLineEdit()
        self.target_column.setPlaceholderText('Target column name (leave empty for image folders)')
        
        # Test/validation split
        split_layout = QHBoxLayout()
        self.test_split = QDoubleSpinBox()
        self.test_split.setRange(0.0, 1.0)
        self.test_split.setValue(0.2)
        self.test_split.setSingleStep(0.05)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.0, 1.0)
        self.val_split.setValue(0.1)
        self.val_split.setSingleStep(0.05)
        
        split_layout.addWidget(QLabel('Test Split:'))
        split_layout.addWidget(self.test_split)
        split_layout.addWidget(QLabel('Validation Split:'))
        split_layout.addWidget(self.val_split)
        split_layout.addStretch()
        
        # Data normalization
        norm_layout = QHBoxLayout()
        self.normalize_check = QCheckBox('Normalize to [0, 1]')
        self.standardize_check = QCheckBox('Standardize (mean=0, std=1)')
        self.normalize_check.setChecked(True)
        
        norm_layout.addWidget(self.normalize_check)
        norm_layout.addWidget(self.standardize_check)
        norm_layout.addStretch()
        
        # Connect checkboxes to be mutually exclusive
        self.normalize_check.toggled.connect(
            lambda: self.standardize_check.setChecked(False) if self.normalize_check.isChecked() else None
        )
        self.standardize_check.toggled.connect(
            lambda: self.normalize_check.setChecked(False) if self.standardize_check.isChecked() else None
        )
        
        # Load data button
        self.load_data_btn = QPushButton('Load Data')
        
        data_layout.addLayout(dataset_layout)
        data_layout.addWidget(self.target_column)
        data_layout.addLayout(split_layout)
        data_layout.addLayout(norm_layout)
        data_layout.addWidget(self.load_data_btn)
        data_group.setLayout(data_layout)
        
        # Data preview
        preview_group = QGroupBox('Data Preview')
        self.data_preview = QTableWidget()
        self.data_preview.setColumnCount(5)
        self.data_preview.setHorizontalHeaderLabels(['Split', 'Shape', 'Min', 'Max', 'Mean'])
        self.data_preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.data_preview)
        preview_group.setLayout(preview_layout)
        
        # Add to main layout
        layout.addWidget(data_group)
        layout.addWidget(preview_group)
        
    def setup_connections(self):
        # Model tab
        self.create_btn.clicked.connect(self.create_model)
        self.visualize_btn.clicked.connect(self.visualize_model)
        
        # Training tab
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.save_btn.clicked.connect(self.save_model)
        self.load_btn.clicked.connect(self.load_model)
        
        # Data tab
        self.load_data_btn.clicked.connect(self.load_data)
        
    def browse_dataset(self):
        # Open file dialog to select dataset
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Dataset',
            '',
            'CSV Files (*.csv);;All Files (*)',
            options=options
        )
        
        if file_path:
            self.dataset_path.setText(file_path)
            
    def create_model(self):
        try:
            logger.info("Creating new model")
            # Get model configuration from UI
            input_size = self.input_size.value()
            hidden_sizes = [int(x.strip()) for x in self.hidden_layers.text().split(',') if x.strip().isdigit()]
            output_size = self.output_size.value()
            activation = self.activation.currentText().lower()
            dropout = self.dropout.value() if self.dropout.value() > 0 else None
            use_batch_norm = self.batch_norm.isChecked()
            
            logger.debug(f"Model configuration: input_size={input_size}, hidden_sizes={hidden_sizes}, "
                       f"output_size={output_size}, activation={activation}, dropout={dropout}, "
                       f"batch_norm={use_batch_norm}")
            
            # Create model
            self.model = NeuralNetwork(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                activation=activation,
                dropout=dropout,
                batch_norm=use_batch_norm
            )
            
            logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
            
            # Update model summary
            self.update_model_summary()
            
            # Enable training controls
            self.train_btn.setEnabled(True)
            self.visualize_btn.setEnabled(True)
            
            logger.info("Model creation completed successfully")
            
        except Exception as e:
            error_msg = f'Failed to create model: {str(e)}'
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, 'Error', error_msg)
    
    def load_data(self):
        """Load and prepare the dataset for training."""
        try:
            logger.info("Loading dataset...")
            
            # Show file dialog to select dataset
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                'Open Dataset',
                '',
                'CSV Files (*.csv);;All Files (*)',
                options=options
            )
            
            if not file_path:
                logger.info("Dataset loading cancelled by user")
                return
                
            logger.info(f"Loading dataset from: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Data preprocessing - handle non-numeric data
            # Convert categorical columns to numeric using label encoding
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Use label encoding for categorical data
                    df[column] = pd.Categorical(df[column]).codes
            
            # Remove any rows with NaN values that might result from conversion
            df = df.dropna()
            
            # Convert all data to float32 for consistency
            df = df.astype(np.float32)
            
            # Assuming the last column is the target and the rest are features
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Get the actual input size from the dataset
            actual_input_size = X.shape[1]
            logger.info(f"Dataset has {actual_input_size} features")
            
            # Store the old input size for notification
            old_input_size = self.input_size.value()
            
            # Update the input size in the UI to match the dataset
            self.input_size.setValue(actual_input_size)
            
            # If a model already exists, recreate it with the correct input size
            if self.model is not None:
                logger.info(f"Recreating model with input size {actual_input_size} to match dataset")
                self.create_model()
                # Show user notification about model recreation
                QMessageBox.information(
                    self, 
                    'Model Updated', 
                    f'The model input size has been automatically updated from {old_input_size} to {actual_input_size} to match your dataset.\n\nThe model has been recreated with the correct dimensions.'
                )
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Split into train and validation sets (80-20 split)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            batch_size = self.batch_size.value()
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Update UI
            self.status_bar.showMessage(f"Loaded dataset with {len(dataset)} samples", 3000)
            logger.info(f"Dataset loaded successfully with {len(dataset)} samples")
            
            # Enable training button if model exists
            if self.model is not None:
                self.train_btn.setEnabled(True)
            
            # Update data preview
            self.update_data_preview(df, X, y)
            
        except Exception as e:
            error_msg = f'Failed to load dataset: {str(e)}'
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, 'Error', error_msg)
    
    def update_data_preview(self, df, X, y):
        """Update the data preview table with dataset statistics."""
        try:
            logger.info("Updating data preview")
            
            # Clear existing data
            self.data_preview.setRowCount(0)
            
            # Calculate statistics for full dataset
            full_row = self.data_preview.rowCount()
            self.data_preview.insertRow(full_row)
            self.data_preview.setItem(full_row, 0, QTableWidgetItem("Full Dataset"))
            self.data_preview.setItem(full_row, 1, QTableWidgetItem(f"{X.shape[0]} samples, {X.shape[1]} features"))
            self.data_preview.setItem(full_row, 2, QTableWidgetItem(f"{X.min():.4f}"))
            self.data_preview.setItem(full_row, 3, QTableWidgetItem(f"{X.max():.4f}"))
            self.data_preview.setItem(full_row, 4, QTableWidgetItem(f"{X.mean():.4f}"))
            
            # Calculate statistics for training set (80% of data)
            train_size = int(0.8 * len(X))
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            train_row = self.data_preview.rowCount()
            self.data_preview.insertRow(train_row)
            self.data_preview.setItem(train_row, 0, QTableWidgetItem("Training Set"))
            self.data_preview.setItem(train_row, 1, QTableWidgetItem(f"{X_train.shape[0]} samples, {X_train.shape[1]} features"))
            self.data_preview.setItem(train_row, 2, QTableWidgetItem(f"{X_train.min():.4f}"))
            self.data_preview.setItem(train_row, 3, QTableWidgetItem(f"{X_train.max():.4f}"))
            self.data_preview.setItem(train_row, 4, QTableWidgetItem(f"{X_train.mean():.4f}"))
            
            # Calculate statistics for validation set (20% of data)
            X_val = X[train_size:]
            y_val = y[train_size:]
            
            val_row = self.data_preview.rowCount()
            self.data_preview.insertRow(val_row)
            self.data_preview.setItem(val_row, 0, QTableWidgetItem("Validation Set"))
            self.data_preview.setItem(val_row, 1, QTableWidgetItem(f"{X_val.shape[0]} samples, {X_val.shape[1]} features"))
            self.data_preview.setItem(val_row, 2, QTableWidgetItem(f"{X_val.min():.4f}"))
            self.data_preview.setItem(val_row, 3, QTableWidgetItem(f"{X_val.max():.4f}"))
            self.data_preview.setItem(val_row, 4, QTableWidgetItem(f"{X_val.mean():.4f}"))
            
            # Add target variable statistics
            target_row = self.data_preview.rowCount()
            self.data_preview.insertRow(target_row)
            self.data_preview.setItem(target_row, 0, QTableWidgetItem("Target Variable"))
            self.data_preview.setItem(target_row, 1, QTableWidgetItem(f"{len(y)} samples"))
            self.data_preview.setItem(target_row, 2, QTableWidgetItem(f"{y.min():.4f}"))
            self.data_preview.setItem(target_row, 3, QTableWidgetItem(f"{y.max():.4f}"))
            self.data_preview.setItem(target_row, 4, QTableWidgetItem(f"{y.mean():.4f}"))
            
            # Add class distribution if it's a classification problem
            unique_classes = np.unique(y)
            if len(unique_classes) <= 10:  # Only show for classification with reasonable number of classes
                for i, class_label in enumerate(unique_classes):
                    class_count = np.sum(y == class_label)
                    class_percentage = (class_count / len(y)) * 100
                    
                    class_row = self.data_preview.rowCount()
                    self.data_preview.insertRow(class_row)
                    self.data_preview.setItem(class_row, 0, QTableWidgetItem(f"Class {int(class_label)}"))
                    self.data_preview.setItem(class_row, 1, QTableWidgetItem(f"{class_count} samples ({class_percentage:.1f}%)"))
                    self.data_preview.setItem(class_row, 2, QTableWidgetItem("-"))
                    self.data_preview.setItem(class_row, 3, QTableWidgetItem("-"))
                    self.data_preview.setItem(class_row, 4, QTableWidgetItem("-"))
            
            logger.info("Data preview updated successfully")
            
        except Exception as e:
            error_msg = f'Failed to update data preview: {str(e)}'
            logger.error(error_msg, exc_info=True)
            # Don't show error dialog for preview update as it's not critical
    
    def visualize_model(self):
        """Visualize the neural network architecture."""
        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'Please create a model first')
            return
            
        try:
            logger.info("Visualizing model architecture")
            
            # Create a simple text representation of the model
            model_str = str(self.model)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Create a dialog to show the model architecture
            dialog = QDialog(self)
            dialog.setWindowTitle('Model Visualization')
            dialog.setMinimumSize(600, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Add model summary
            summary_label = QLabel('Model Architecture:')
            summary_text = QTextEdit()
            summary_text.setReadOnly(True)
            summary_text.setPlainText(model_str)
            
            # Add parameter count
            params_label = QLabel(f'Total Trainable Parameters: {total_params:,}')
            font = params_label.font()
            font.setBold(True)
            params_label.setFont(font)
            
            # Add close button
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(dialog.accept)
            
            # Add widgets to layout
            layout.addWidget(summary_label)
            layout.addWidget(summary_text)
            layout.addWidget(params_label)
            layout.addWidget(button_box)
            
            dialog.exec()
            
        except Exception as e:
            error_msg = f'Failed to visualize model: {str(e)}'
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, 'Error', error_msg)
    
    def update_model_summary(self):
        if self.model is None:
            self.model_summary.setPlainText('No model created yet')
            return
            
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Create summary
        summary = []
        summary.append(f'Model Architecture:')
        summary.append('-' * 50)
        
        # Add layers
        for name, param in self.model.named_parameters():
            summary.append(f'{name}: {tuple(param.shape)}')
            
        summary.append('-' * 50)
        summary.append(f'Total parameters: {total_params:,}')
        summary.append(f'Trainable parameters: {trainable_params:,}')
        
        self.model_summary.setPlainText('\n'.join(summary))
        
    def start_training(self):
        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'Please create a model first')
            logger.warning("Attempted to start training without a model")
            return
            
        if self.train_loader is None:
            QMessageBox.warning(self, 'Warning', 'Please load training data first')
            logger.warning("Attempted to start training without training data")
            return
            
        try:
            logger.info("Starting training process")
            
            # Disable UI elements during training
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress.setValue(0)
            self.training_log.clear()
            
            # Log training configuration
            logger.info(f"Training configuration: epochs={self.epochs.value()}, "
                       f"learning_rate={self.learning_rate.value()}, "
                       f"optimizer={self.optimizer.currentText().lower()}, "
                       f"weight_decay={self.weight_decay.value()}, "
                       f"loss_fn={self.loss_fn.currentText().lower()}, "
                       f"device={self.device.currentText().lower()}")
            
            # Create and start training thread
            self.training_thread = TrainingThread(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=self.epochs.value(),
                learning_rate=self.learning_rate.value(),
                optimizer=self.optimizer.currentText().lower(),
                weight_decay=self.weight_decay.value(),
                loss_fn=self.loss_fn.currentText().lower(),
                device=self.device.currentText().lower()
            )
            
            # Connect signals
            self.training_thread.progress_updated.connect(self.update_training_progress)
            self.training_thread.log_message.connect(self.log_message)
            self.training_thread.training_finished.connect(self.training_finished)
            self.training_thread.training_error.connect(self.training_error)
            
            # Start training
            logger.debug("Starting training thread")
            self.training_thread.start()
            logger.info("Training started successfully")
            
        except Exception as e:
            error_msg = f'Failed to start training: {str(e)}'
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, 'Error', error_msg)
            
    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            logger.info("Stopping training...")
            self.training_thread.stop()
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            logger.info("Training stopped by user request")
    
    def update_training_progress(self, epoch, train_loss, val_loss):
        # Update progress bar
        progress = int((epoch / self.epochs.value()) * 100)
        self.progress.setValue(progress)
        
        # Update status
        status = f'Epoch {epoch}/{self.epochs.value()} - Loss: {train_loss:.4f}'
        if val_loss is not None and val_loss > 0:
            status += f' - Val Loss: {val_loss:.4f}'
        self.status_bar.showMessage(status)
    
    def log_message(self, message):
        try:
            self.training_log.append(message)
            # Auto-scroll to bottom
            scrollbar = self.training_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            logger.error(f"Failed to update log message: {str(e)}", exc_info=True)
    
    def training_finished(self, history):
        try:
            logger.info("Training completed successfully")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress.setValue(100)
            self.log_message('Training finished!')
            
            # Log final metrics
            if history and 'train_loss' in history and history['train_loss']:
                final_train_loss = history['train_loss'][-1]
                logger.info(f"Final training loss: {final_train_loss:.6f}")
                if 'val_loss' in history and history['val_loss']:
                    final_val_loss = history['val_loss'][-1]
                    logger.info(f"Final validation loss: {final_val_loss:.6f}")
            
            # Plot training history
            self.plot_training_history(history)
        except Exception as e:
            error_msg = f"Error in training_finished: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.warning(self, 'Warning', error_msg)
    
    def training_error(self, error_msg):
        logger.error(f"Training error: {error_msg}")
        QMessageBox.critical(self, 'Training Error', error_msg)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def plot_training_history(self, history):
        try:
            # Create a new window for the plot
            plot_window = QMainWindow(self)
            plot_window.setWindowTitle('Training History')
            plot_window.setGeometry(200, 200, 1000, 600)
            
            # Create figure and canvas
            fig = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvas(fig)
            
            # Create subplots
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            # Plot training and validation loss
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
            
            if 'val_loss' in history and history['val_loss']:
                ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            
            ax1.set_title('Training & Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot metrics if available
            metrics = [k.replace('train_', '') for k in history.keys() 
                      if k.startswith('train_') and k != 'train_loss']
            
            for metric in metrics:
                train_metric = f'train_{metric}'
                val_metric = f'val_{metric}'
                
                if train_metric in history:
                    ax2.plot(epochs, history[train_metric], 'b--', label=f'Training {metric}')
                if val_metric in history:
                    ax2.plot(epochs, history[val_metric], 'r--', label=f'Validation {metric}')
            
            if metrics:
                ax2.set_title('Training & Validation Metrics')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Metric')
                ax2.legend()
                ax2.grid(True)
            
            # Adjust layout
            fig.tight_layout()
            
            # Set layout
            plot_window.setCentralWidget(canvas)
            plot_window.show()
            
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f'Failed to plot training history: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def save_model(self):
        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'No model to save')
            return
            
        try:
            # Open file dialog to select save location
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                'Save Model',
                '',
                'PyTorch Model (*.pth);;All Files (*)',
                options=options
            )
            
            if not file_path:
                return  # User cancelled
                
            # Ensure file has .pth extension
            if not file_path.endswith('.pth'):
                file_path += '.pth'
            
            # Get model configuration
            config = {
                'input_size': self.input_size.value(),
                'hidden_sizes': [int(x.strip()) for x in self.hidden_layers.text().split(',') if x.strip().isdigit()],
                'output_size': self.output_size.value(),
                'activation': self.activation.currentText().lower(),
                'dropout': self.dropout.value() if self.dropout.value() > 0 else None,
                'batch_norm': self.batch_norm.isChecked()
            }
            
            # Save the model
            save_model(
                model=self.model,
                path=file_path,
                config=config
            )
            
            self.status_bar.showMessage(f'Model saved to {file_path}', 5000)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save model: {str(e)}')
    
    def load_model(self):
        try:
            # Open file dialog to select model file
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                'Load Model',
                '',
                'PyTorch Model (*.pth);;All Files (*)',
                options=options
            )
            
            if not file_path:
                return  # User cancelled
                
            # Load the model
            result = load_model(
                path=file_path,
                model_class=NeuralNetwork
            )
            
            self.model = result['model']
            
            # Update UI with loaded model configuration
            if 'config' in result and result['config']:
                config = result['config']
                self.input_size.setValue(config.get('input_size', 784))
                
                hidden_sizes = config.get('hidden_sizes', [128, 64])
                self.hidden_layers.setText(', '.join(map(str, hidden_sizes)))
                
                self.output_size.setValue(config.get('output_size', 10))
                
                activation = config.get('activation', 'relu')
                index = self.activation.findText(activation, Qt.MatchFixedString)
                if index >= 0:
                    self.activation.setCurrentIndex(index)
                
                dropout = config.get('dropout', 0.0)
                self.dropout.setValue(dropout if dropout is not None else 0.0)
                
                self.batch_norm.setChecked(config.get('batch_norm', False))
            
            # Update model summary
            self.update_model_summary()
            
            self.status_bar.showMessage(f'Model loaded from {file_path}', 5000)
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model: {str(e)}')
    
    def closeEvent(self, event):
        try:
            logger.info("Application closing...")
            # Stop training if running
            if hasattr(self, 'training_thread') and self.training_thread is not None:
                if self.training_thread.isRunning():
                    logger.info("Stopping training thread...")
                    self.training_thread.stop()
                    self.training_thread.wait()
                    logger.info("Training thread stopped")
            
            # Clean up resources
            if hasattr(self, 'model') and self.model is not None:
                logger.debug("Cleaning up model resources")
                # Add any model cleanup code here
            
            logger.info("Application closed successfully")
            
        except Exception as e:
            logger.error(f"Error during application close: {str(e)}", exc_info=True)
        finally:
            event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application icon
    icon_path = os.path.join(project_root, 'assets', 'logo.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        logger.info(f"Application icon set from: {icon_path}")
    else:
        logger.warning(f"Application icon not found at: {icon_path}")
    
    # Create and show the main window
    window = NeuralNetworkApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
