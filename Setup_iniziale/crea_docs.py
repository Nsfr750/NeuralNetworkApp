"""
Help Documentation and Tooltips

This module provides interactive help and documentation for the Neural Network Application.
It includes tooltips, usage examples, and detailed documentation for all components.
"""

from typing import Dict, List, Optional, Any, Type, Callable, Union
import inspect
import textwrap
from dataclasses import dataclass
from enum import Enum, auto
import webbrowser
import os

# Import components that need documentation
from neuralnetworkapp.builder import NetworkBuilder, LayerType
from neuralnetworkapp.visualization import TrainingVisualizer, RealTimePlot


class DocCategory(Enum):
    """Categories for documentation entries."""
    GETTING_STARTED = auto()
    NETWORK_BUILDER = auto()
    VISUALIZATION = auto()
    TRAINING = auto()
    EXPORT = auto()
    EXAMPLES = auto()
    TROUBLESHOOTING = auto()


@dataclass
class DocumentationEntry:
    """A single documentation entry with title, content, and related items."""
    title: str
    content: str
    category: DocCategory
    related: List[str] = None
    code_example: str = None
    
    def to_html(self) -> str:
        """Convert the documentation entry to HTML format."""
        html = f"""
        <div class="doc-entry">
            <h2>{self.title}</h2>
            <div class="doc-category">Category: {self.category.name.replace('_', ' ').title()}</div>
            <div class="doc-content">
                {self.content}
            </div>
        """
        
        if self.code_example:
            html += f"""
            <div class="code-example">
                <h4>Example:</h4>
                <pre><code>{self.code_example}</code></pre>
            </div>
            """
            
        if self.related:
            related_links = ", ".join(f'<a href="#{r.lower().replace(" ", "-")}">{r}</a>' for r in self.related)
            html += f"""
            <div class="related-topics">
                <h4>Related Topics:</h4>
                <p>{related_links}</p>
            </div>
            """
            
        html += "</div>"
        return textwrap.dedent(html)


class HelpSystem:
    """
    A comprehensive help system for the Neural Network Application.
    
    Features:
    - Interactive help in the console
    - Web-based documentation
    - Tooltips for UI elements
    - Search functionality
    - Code examples
    """
    
    def __init__(self):
        """Initialize the help system with documentation entries."""
        self.docs: Dict[str, DocumentationEntry] = {}
        self._setup_documentation()
    
    def _setup_documentation(self) -> None:
        """Set up all documentation entries."""
        # Getting Started
        self._add_doc(
            "Getting Started",
            """
            # Getting Started with Neural Network Application
            
            Welcome to the Neural Network Application! This guide will help you get started
            with building, training, and deploying neural networks.
            
            ## Key Features:
            - Intuitive network architecture builder
            - Real-time training visualization
            - Support for various neural network architectures
            - Transfer learning capabilities
            - Model export to ONNX and TensorFlow Lite
            
            ## Quick Start:
            1. Create a new network using the NetworkBuilder
            2. Train your model with real-time monitoring
            3. Evaluate and export your model for deployment
            """,
            DocCategory.GETTING_STARTED,
            ["NetworkBuilder", "Training", "Export"]
        )
        
        # NetworkBuilder Documentation
        self._add_doc(
            "NetworkBuilder",
            """
            # NetworkBuilder
            
            The `NetworkBuilder` class provides an intuitive interface for building
            neural network architectures with real-time visualization.
            
            ## Key Features:
            - Add various layer types (convolutional, linear, activation, etc.)
            - Automatic shape inference
            - Visualize network architecture
            - Export to PyTorch model
            - Import/export network configuration as JSON
            
            ## Usage:
            ```python
            from neuralnetworkapp.builderimport NetworkBuilder, LayerType
            
            # Create a new network
            builder = NetworkBuilder(input_shape=(3, 32, 32))
            
            # Add layers
            (builder
             .add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)
             .add_layer(LayerType.RELU)
             .add_layer(LayerType.MAXPOOL2D, kernel_size=2)
             .add_layer(LayerType.FLATTEN)
             .add_layer(LayerType.LINEAR, out_features=10))
            
            # Visualize the network
            builder.visualize("network_architecture.png")
            
            # Build the PyTorch model
            model = builder.build()
            ```
            """,
            DocCategory.NETWORK_BUILDER,
            ["Layer Types", "Visualization", "Export"]
        )
        
        # Layer Types Documentation
        self._add_doc(
            "Layer Types",
            """
            # Supported Layer Types
            
            The following layer types are supported in the NetworkBuilder:
            
            ## Convolutional Layers
            - `CONV2D`: 2D convolution layer
            - `MAXPOOL2D`: 2D max pooling
            - `AVGPOOL2D`: 2D average pooling
            - `ADAPTIVEAVGPOOL2D`: 2D adaptive average pooling
            
            ## Linear Layers
            - `LINEAR`: Fully connected layer
            - `FLATTEN`: Flattens the input
            
            ## Normalization
            - `BATCHNORM2D`: Batch normalization
            - `DROPOUT`: Dropout layer
            
            ## Activation Functions
            - `RELU`: Rectified Linear Unit
            - `LEAKYRELU`: Leaky ReLU
            - `SIGMOID`: Sigmoid activation
            - `TANH`: Hyperbolic tangent
            - `SOFTMAX`: Softmax activation
            
            ## Special
            - `IDENTITY`: Pass-through layer
            - `RESIDUAL`: Residual connection block
            """,
            DocCategory.NETWORK_BUILDER,
            ["NetworkBuilder", "Examples"]
        )
        
        # Visualization Documentation
        self._add_doc(
            "Visualization",
            """
            # Training Visualization
            
            The `TrainingVisualizer` class provides real-time visualization of training metrics,
            model architecture, and feature spaces.
            
            ## Features:
            - Live updating plots of loss and metrics
            - Support for multiple metrics and datasets (train/val/test)
            - TensorBoard integration
            - Feature space visualization (t-SNE, PCA, UMAP)
            - Attention map visualization
            
            ## Usage:
            ```python
            from neuralnetworkapp.visualizationimport TrainingVisualizer
            
            # Create a visualizer
            visualizer = TrainingVisualizer(log_dir='runs/experiment_1')
            
            # During training:
            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Training code...
                    
                    # Log metrics
                    visualizer.add_scalar('train/loss', loss.item(), global_step)
                    visualizer.add_scalar('train/accuracy', accuracy, global_step)
                    
                    # Update plots every N steps
                    if batch_idx % 10 == 0:
                        visualizer.plot_metrics()
            
            # Visualize embeddings
            visualizer.visualize_embedding(
                embeddings=features,
                labels=labels,
                method='tsne',  # 'pca', 'tsne', or 'umap'
                title='Feature Space Visualization'
            )
            
            # Close the visualizer when done
            visualizer.close()
            ```
            """,
            DocCategory.VISUALIZATION,
            ["Training", "Examples"]
        )
        
        # Training Documentation
        self._add_doc(
            "Training",
            """
            # Model Training
            
            This section covers how to train your neural network models using the provided utilities.
            
            ## Training Process:
            1. **Prepare your data** using PyTorch DataLoaders
            2. **Define your model** using NetworkBuilder or custom PyTorch modules
            3. **Set up the trainer** with your model, loss function, and optimizer
            4. **Start training** with real-time monitoring
            5. **Evaluate** on validation/test sets
            
            ## Example Training Loop:
            ```python
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from neuralnetworkapp.builderimport NetworkBuilder
            from neuralnetworkapp.visualizationimport TrainingVisualizer
            
            # 1. Create model
            builder = NetworkBuilder(input_shape=(3, 32, 32))
            builder.add_layer(LayerType.CONV2D, out_channels=32, kernel_size=3, padding=1)\
                   .add_layer(LayerType.RELU)\
                   .add_layer(LayerType.FLATTEN)\
                   .add_layer(LayerType.LINEAR, out_features=10)
            model = builder.build()
            
            # 2. Set up loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 3. Create visualizer
            visualizer = TrainingVisualizer()
            
            # 4. Training loop
            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    # Log metrics
                    visualizer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
                    
                    # Update plots
                    if batch_idx % 10 == 0:
                        visualizer.plot_metrics()
            
            # 5. Clean up
            visualizer.close()
            ```
            """,
            DocCategory.TRAINING,
            ["NetworkBuilder", "Visualization", "Transfer Learning"]
        )
        
        # Export Documentation
        self._add_doc(
            "Export",
            """
            # Model Export
            
            Export your trained models to various formats for deployment.
            
            ## Supported Formats:
            - **ONNX**: Open Neural Network Exchange format
            - **TensorFlow Lite**: For mobile and embedded devices
            - **TorchScript**: For production deployment with PyTorch
            
            ## Export Example:
            ```python
            import torch
            from neuralnetworkapp.builderimport NetworkBuilder
            
            # Create and train a model
            builder = NetworkBuilder(input_shape=(3, 32, 32))
            # ... add layers ...
            model = builder.build()
            
            # Export to ONNX
            dummy_input = torch.randn(1, 3, 32, 32)
            torch.onnx.export(
                model,
                dummy_input,
                "model.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Export to TensorFlow Lite (requires onnx-tf)
            # See transfer_learning.py for implementation
            ```
            """,
            DocCategory.EXPORT,
            ["NetworkBuilder", "Transfer Learning"]
        )
        
        # Transfer Learning Documentation
        self._add_doc(
            "Transfer Learning",
            """
            # Transfer Learning
            
            Leverage pre-trained models for your specific tasks with transfer learning.
            
            ## Available Pre-trained Models:
            - ResNet variants (18, 34, 50, 101, 152)
            - VGG (11, 13, 16, 19) with and without batch normalization
            - DenseNet (121, 169, 201)
            - MobileNetV2 and MobileNetV3
            - EfficientNet (B0-B7)
            - RegNet variants
            
            ## Transfer Learning Example:
            ```python
            from neuralnetworkapp.transferimport get_pretrained_model, fine_tune_model
            
            # Load a pre-trained model
            model = get_pretrained_model(
                model_name='resnet50',
                num_classes=10,  # Number of output classes
                pretrained=True,  # Use pre-trained weights
                freeze_features=True  # Freeze feature extractor
            )
            
            # Fine-tune the model
            history = fine_tune_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=torch.nn.CrossEntropyLoss(),
                num_epochs=10,
                freeze_epochs=5,  # Number of epochs with frozen features
                unfreeze_layers=['layer4', 'fc']  # Layers to unfreeze after freeze_epochs
            )
            
            # Export the fine-tuned model
            torch.save(model.state_dict(), 'fine_tuned_model.pth')
            ```
            """,
            DocCategory.TRAINING,
            ["Training", "Export"]
        )
        
        # Examples
        self._add_doc(
            "Examples",
            """
            # Example Projects
            
            Here are some example projects you can build with this library:
            
            ## 1. Image Classification
            - CIFAR-10/100 classification
            - Custom image classification
            - Transfer learning with pre-trained models
            
            ## 2. Object Detection
            - Fine-tune Faster R-CNN or RetinaNet
            - Custom object detection
            
            ## 3. Semantic Segmentation
            - U-Net for medical image segmentation
            - DeepLab for scene parsing
            
            ## 4. Generative Models
            - GANs for image generation
            - Variational Autoencoders
            
            Check the `examples/` directory for complete code examples.
            """,
            DocCategory.EXAMPLES,
            ["Getting Started", "Training"]
        )
        
        # Troubleshooting
        self._add_doc(
            "Troubleshooting",
            """
            # Troubleshooting Guide
            
            Common issues and solutions:
            
            ## 1. Out of Memory (OOM) Errors
            - Reduce batch size
            - Use gradient accumulation
            - Enable mixed precision training
            - Use a smaller model
            
            ## 2. Training is Slow
            - Enable CUDA if available
            - Use DataLoader with multiple workers
            - Reduce model complexity
            - Use larger batch sizes
            
            ## 3. Poor Model Performance
            - Check your data preprocessing
            - Try different learning rates
            - Add more training data
            - Try data augmentation
            - Use transfer learning
            
            ## 4. Installation Issues
            - Make sure you have the required dependencies
            - Use a virtual environment
            - Check PyTorch and CUDA compatibility
            """,
            DocCategory.TROUBLESHOOTING,
            ["Getting Started"]
        )
    
    def _add_doc(self, title: str, content: str, category: DocCategory,
                related: List[str] = None, code_example: str = None) -> None:
        """Add a documentation entry."""
        # Format the content (remove common indentation)
        content = textwrap.dedent(content.lstrip('\n')).strip()
        
        # Format code example if provided
        if code_example:
            code_example = textwrap.dedent(code_example).strip()
        
        self.docs[title.lower()] = DocumentationEntry(
            title=title,
            content=content,
            category=category,
            related=related or [],
            code_example=code_example
        )
    
    def get_doc(self, title: str) -> Optional[DocumentationEntry]:
        """Get a documentation entry by title (case-insensitive)."""
        return self.docs.get(title.lower())
    
    def search(self, query: str, category: Optional[DocCategory] = None) -> List[DocumentationEntry]:
        """Search documentation entries for a query."""
        query = query.lower()
        results = []
        
        for doc in self.docs.values():
            if (query in doc.title.lower() or 
                query in doc.content.lower() or
                any(query in r.lower() for r in doc.related)):
                
                if category is None or doc.category == category:
                    results.append(doc)
        
        # Sort by relevance (simple: title match > content match > related match)
        def relevance(doc):
            score = 0
            if query in doc.title.lower():
                score += 2
            if query in doc.content.lower():
                score += 1
            return -score  # Sort in descending order
        
        results.sort(key=relevance)
        return results
    
    def show_help(self, topic: str = None) -> None:
        """Display help for a specific topic or list available topics."""
        if topic is None:
            self._list_topics()
            return
        
        doc = self.get_doc(topic)
        if doc is None:
            print(f"No help found for '{topic}'. Showing available topics:")
            self._list_topics()
            return
        
        print(f"\n{'='*80}")
        print(f"{doc.title}")
        print(f"{'='*80}")
        print(doc.content)
        
        if doc.related:
            print("\nRelated Topics:")
            for related in doc.related:
                print(f"- {related}")
    
    def _list_topics(self) -> None:
        """List all available help topics."""
        print("\nAvailable Help Topics:")
        print("-" * 40)
        
        # Group by category
        by_category = {}
        for doc in self.docs.values():
            if doc.category not in by_category:
                by_category[doc.category] = []
            by_category[doc.category].append(doc.title)
        
        # Print by category
        for category, titles in by_category.items():
            print(f"\n{category.name.replace('_', ' ').title()}:")
            for title in sorted(titles):
                print(f"  {title}")
        
        print("\nType 'help(topic)' to get help on a specific topic.")
    
    def generate_html_docs(self, output_dir: str = 'docs') -> None:
        """Generate HTML documentation."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create index.html
        with open(os.path.join(output_dir, 'index.html'), 'w') as f:
            f.write(self._generate_index_html())
        
        # Create individual pages
        for doc in self.docs.values():
            filename = f"{doc.title.lower().replace(' ', '-')}.html"
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(self._generate_doc_page(doc))
        
        # Create CSS
        with open(os.path.join(output_dir, 'style.css'), 'w') as f:
            f.write(self._generate_css())
        
        print(f"Documentation generated in '{output_dir}'. Open 'index.html' in a web browser.")
    
    def _generate_index_html(self) -> str:
        """Generate the index.html file."""
        # Group by category
        by_category = {}
        for doc in self.docs.values():
            if doc.category not in by_category:
                by_category[doc.category] = []
            by_category[doc.category].append(doc)
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Neural Network Application - Documentation</title>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <header>
                <h1>Neural Network Application</h1>
                <p>Comprehensive Documentation</p>
            </header>
            
            <div class="container">
                <nav class="sidebar">
                    <h2>Table of Contents</h2>
                    <ul>
        """
        
        # Add TOC
        for category, docs in by_category.items():
            html += f'<li class="category">{category.name.replace("_", " ").title()}</li>\n'
            for doc in sorted(docs, key=lambda d: d.title):
                filename = f"{doc.title.lower().replace(' ', '-')}.html"
                html += f'<li><a href="{filename}">{doc.title}</a></li>\n'
        html += """
                    </ul>
                </nav>
                
                <main>
                    <h2>Welcome to the Neural Network Application</h2>
                    <p>This documentation provides comprehensive information about using the Neural Network Application.</p>
                    
                    <h3>Getting Started</h3>
                    <p>New to the Neural Network Application? Start with the <a href="getting-started.html">Getting Started</a> guide.</p>
                    
                    <h3>Quick Links</h3>
                    <ul>
                        <li><a href="networkbuilder.html">NetworkBuilder</a> - Build neural network architectures</li>
                        <li><a href="visualization.html">Visualization</a> - Monitor training progress</li>
                        <li><a href="training.html">Training</a> - Train your models</li>
                        <li><a href="export.html">Export</a> - Export models for deployment</li>
                    </ul>
                </main>
            </div>
            
            <footer>
                <p>&copy; 2023 Neural Network Application. All rights reserved.</p>
            </footer>
        </body>
        </html>
        """
        
        return textwrap.dedent(html)
    
    def _generate_doc_page(self, doc: DocumentationEntry) -> str:
        """Generate an HTML page for a documentation entry."""
        # Convert markdown to HTML (simplified)
        content = doc.content
        
        # Simple markdown to HTML conversion
        content = content.replace('\n\n', '</p>\n<p>')
        content = content.replace('\n', '<br>\n')
        
        # Headers
        for i in range(3, 0, -1):
            content = content.replace('#' * i + ' ', f'<h{i}>')
            content = content.replace('\n' + '#' * i, f'</h{i}>\n')
        
        # Code blocks
        while '```' in content:
            content = content.replace('```python', '<pre><code class="language-python">', 1)
            content = content.replace('```', '</code></pre>', 1)
        
        # Inline code
        content = content.replace('`', '<code>')
        
        # Lists
        content = content.replace('- ', '<li>')
        content = content.replace('\n<li>', '</li>\n<li>')
        
        # Prepare examples section
        examples_html = ''
        if doc.code_example:
            examples_html = f'''
            <section id="examples">
                <h2>Examples</h2>
                <pre><code class="language-python">{doc.code_example}</code></pre>
            </section>'''
        
        # Prepare related links section
        related_links = ''
        if doc.related:
            related_links = '\n'.join(
                f'<li><a href="{r.lower().replace(" ", "-")}.html">{r}</a></li>'
                for r in doc.related
            )
        
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title} - Neural Network Application</title>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <header>
                <h1>{title}</h1>
                <nav>
                    <a href="index.html">Home</a>
                </nav>
            </header>
            
            <div class="container">
                <nav class="sidebar">
                    <h3>In This Section</h3>
                    <ul>
                        <li><a href="#overview">Overview</a></li>
                        <li><a href="#usage">Usage</a></li>
                        <li><a href="#examples">Examples</a></li>
                    </ul>
                </nav>
                
                <main>
                    <section id="overview">
                        <h2>Overview</h2>
                        {content}
                    </section>
                    
                    {examples}
                    
                    <section class="related">
                        <h2>Related Topics</h2>
                        <ul>
                            {related}
                        </ul>
                    </section>
                </main>
            </div>
            
            <footer>
                <p>&copy; 2023 Neural Network Application. All rights reserved.</p>
            </footer>
        </body>
        </html>
        """.format(
            title=doc.title,
            content=content,
            examples=examples_html,
            related=related_links
        )
        
        return textwrap.dedent(html)
    
    def _generate_css(self) -> str:
        """Generate CSS for the documentation."""
        return """
        /* Base Styles */
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #6c757d;
            --background-color: #f8f9fa;
            --text-color: #212529;
            --border-color: #dee2e6;
            --code-bg: #f8f9fa;
            --code-color: #e83e8c;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            padding: 0;
            margin: 0;
        }
        
        /* Layout */
        .container {
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .sidebar {
            width: 250px;
            padding: 20px;
            background-color: white;
            border-right: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
        }
        
        main {
            flex: 1;
            padding: 20px 40px;
            background-color: white;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
            margin: 1.5em 0 0.5em;
            line-height: 1.2;
        }
        
        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.75rem; }
        h4 { font-size: 1.5rem; }
        
        p {
            margin-bottom: 1em;
        }
        
        a {
            color: var(--primary-color);
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Code */
        pre, code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            background-color: var(--code-bg);
            border-radius: 3px;
        }
        
        pre {
            padding: 16px;
            overflow: auto;
            line-height: 1.45;
            margin: 1em 0;
            border-left: 4px solid var(--primary-color);
        }
        
        code {
            padding: 0.2em 0.4em;
            font-size: 0.9em;
            color: var(--code-color);
        }
        
        pre code {
            padding: 0;
            background: transparent;
            color: inherit;
        }
        
        /* Navigation */
        nav {
            margin-bottom: 2em;
        }
        
        nav ul {
            list-style: none;
            padding: 0;
        }
        
        nav li {
            margin-bottom: 0.5em;
        }
        
        .category {
            font-weight: bold;
            margin-top: 1em;
            color: var(--secondary-color);
        }
        
        /* Header & Footer */
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 2rem 20px;
            margin-bottom: 2em;
        }
        
        header h1 {
            color: white;
            margin: 0;
        }
        
        footer {
            text-align: center;
            padding: 2em 0;
            margin-top: 2em;
            border-top: 1px solid var(--border-color);
            color: var(--secondary-color);
            font-size: 0.9em;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                height: auto;
                position: relative;
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }
            
            main {
                padding: 20px;
            }
        }
        """


# Create a global help system instance
help_system = HelpSystem()

# Add convenience functions
def help(topic: str = None) -> None:
    """Show help for a specific topic or list available topics."""
    help_system.show_help(topic)


def generate_docs(output_dir: str = 'docs') -> None:
    """Generate HTML documentation."""
    help_system.generate_html_docs(output_dir)


# Example usage
if __name__ == "__main__":
    # Generate documentation
    generate_docs()
    
    # Open in default web browser
    import webbrowser
    webbrowser.open('docs/index.html')
