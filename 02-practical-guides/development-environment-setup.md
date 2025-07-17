# Development Environment Setup

Step-by-step guide to setting up your AI/ML development environment from scratch.

## 🎯 Choose Your Path

### Path 1: Cloud-First (Recommended for Beginners)
**Pros**: No setup, free GPU, all libraries pre-installed  
**Cons**: Internet required, session limits  
**Best for**: Learning, experimentation, small projects

### Path 2: Local Development
**Pros**: Full control, no session limits, works offline  
**Cons**: Setup complexity, hardware requirements  
**Best for**: Serious development, large projects, privacy needs

### Path 3: Hybrid Approach
**Pros**: Best of both worlds, flexible  
**Cons**: More complex workflow  
**Best for**: Professional development, varied project needs

---

## Path 1: Cloud-First Setup (15 minutes)

### Google Colab Setup
1. **Go to Google Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Sign in** with your Google account
3. **Create a new notebook**: File → New notebook
4. **Test your setup**:
   ```python
   # Run this in a cell to test everything works
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.datasets import load_iris
   
   # Load sample data
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   
   # Create a simple plot
   plt.figure(figsize=(8, 6))
   plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'])
   plt.xlabel('Sepal Length')
   plt.ylabel('Sepal Width')
   plt.title('Iris Dataset - Your Setup Works!')
   plt.show()
   
   print("🎉 Your environment is ready!")
   ```

### Enable GPU (Free!)
1. **Runtime** → **Change runtime type**
2. **Hardware accelerator** → **GPU**
3. **Save**

### Pro Tips for Colab:
- Mount Google Drive for persistent storage: `from google.colab import drive; drive.mount('/content/drive')`
- Install additional packages: `!pip install package-name`
- Upload files: Use the folder icon in the sidebar
- Save notebooks to GitHub: File → Save a copy in GitHub

---

## Path 2: Local Development Setup

### Step 1: Install Python (20 minutes)

#### Windows:
```bash
# Download Python from python.org or use Microsoft Store
# Ensure "Add to PATH" is checked during installation

# Verify installation
python --version
pip --version
```

#### macOS:
```bash
# Using Homebrew (recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python

# Or download from python.org
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Step 2: Set Up Virtual Environment (10 minutes)

```bash
# Create a virtual environment
python -m venv ai-ml-env

# Activate it
# Windows:
ai-ml-env\Scripts\activate

# macOS/Linux:
source ai-ml-env/bin/activate

# You should see (ai-ml-env) in your prompt
```

### Step 3: Install Core Packages (15 minutes)

```bash
# Upgrade pip first
pip install --upgrade pip

# Install core data science packages
pip install numpy pandas matplotlib seaborn
pip install scikit-learn jupyter ipywidgets

# Install deep learning frameworks (choose one or both)
pip install tensorflow  # Google's framework
pip install torch torchvision  # Facebook's framework

# Install additional useful packages
pip install plotly bokeh altair  # Advanced visualization
pip install requests beautifulsoup4  # Web scraping
pip install streamlit  # Web apps
pip install black flake8  # Code formatting
```

### Step 4: Install and Configure VS Code (15 minutes)

1. **Download VS Code**: [code.visualstudio.com](https://code.visualstudio.com)
2. **Install Python Extension**: 
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python" by Microsoft
   - Install it
3. **Configure Python Interpreter**:
   - Open Command Palette (Ctrl+Shift+P)
   - Type "Python: Select Interpreter"
   - Choose your virtual environment

### Step 5: Test Your Setup (5 minutes)

Create a file called `test_setup.py`:

```python
#!/usr/bin/env python3
"""Test script to verify AI/ML environment setup."""

import sys
import importlib

# Required packages
packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn',
    'sklearn', 'jupyter', 'tensorflow', 'torch'
]

print("🔍 Testing AI/ML Environment Setup")
print("=" * 40)

# Check Python version
print(f"Python version: {sys.version}")
print()

# Test package imports
failed_imports = []
for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package:<12} - v{version}")
    except ImportError:
        print(f"❌ {package:<12} - NOT INSTALLED")
        failed_imports.append(package)

print()
if failed_imports:
    print(f"❌ Failed imports: {', '.join(failed_imports)}")
    print("Run: pip install " + " ".join(failed_imports))
else:
    print("🎉 All packages installed successfully!")
    
    # Quick functionality test
    print("\n🧪 Quick functionality test:")
    import numpy as np
    import pandas as pd
    
    # Create sample data
    data = np.random.randn(100, 2)
    df = pd.DataFrame(data, columns=['x', 'y'])
    print(f"   Created DataFrame with shape: {df.shape}")
    print(f"   Mean values: x={df.x.mean():.2f}, y={df.y.mean():.2f}")
    print("   ✅ NumPy and Pandas working correctly!")
```

Run it:
```bash
python test_setup.py
```

---

## Path 3: Hybrid Setup

### Local Environment + Cloud Compute

1. **Set up local environment** (following Path 2)
2. **Use cloud for heavy compute**:
   - Train models in Colab/Kaggle
   - Develop and prototype locally
   - Use version control (Git) to sync

### Workflow Example:
```bash
# Local development
git clone your-project
cd your-project
source ai-ml-env/bin/activate
jupyter notebook  # Develop locally

# When ready for heavy compute
git push  # Push to GitHub
# Open in Colab, mount GitHub repo
# Train model in cloud
# Save results back to GitHub
```

---

## Advanced Setup Options

### Docker Environment (Reproducible)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

Build and run:
```bash
docker build -t ai-ml-env .
docker run -p 8888:8888 -v $(pwd):/app ai-ml-env
```

### Conda Environment (Package Management)

```bash
# Install Miniconda/Anaconda
# Download from conda.io

# Create environment
conda create -n ai-ml python=3.9
conda activate ai-ml

# Install packages
conda install numpy pandas matplotlib scikit-learn jupyter
conda install -c conda-forge tensorflow pytorch
```

### GPU Setup (Local)

#### NVIDIA GPU (CUDA):
```bash
# Install NVIDIA drivers first
# Download CUDA Toolkit from developer.nvidia.com

# Install GPU-enabled TensorFlow
pip install tensorflow[and-cuda]

# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Test GPU:
```python
# TensorFlow
import tensorflow as tf
print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))

# PyTorch
import torch
print("PyTorch GPU available:", torch.cuda.is_available())
```

---

## IDE and Editor Options

### Jupyter Notebook/Lab
```bash
# Start Jupyter Notebook
jupyter notebook

# Or Jupyter Lab (more features)
pip install jupyterlab
jupyter lab
```

### VS Code Extensions
Essential extensions for AI/ML:
- Python (Microsoft)
- Jupyter (Microsoft)
- Python Docstring Generator
- GitLens
- Pylance
- Black Formatter

### PyCharm Professional
Features for AI/ML:
- Built-in notebook support
- Database integration
- Profiler
- Scientific mode

### Alternative Editors
- **Vim/Neovim**: With appropriate plugins
- **Emacs**: With ein (Emacs IPython Notebook)
- **Sublime Text**: With packages
- **Atom**: GitHub's editor (discontinued but still usable)

---

## Troubleshooting Common Issues

### Import Errors
```bash
# Wrong Python/pip version
which python
which pip

# Virtual environment not activated
source ai-ml-env/bin/activate  # Unix
ai-ml-env\Scripts\activate     # Windows

# Package not installed in current environment
pip list | grep package-name
```

### Jupyter Kernel Issues
```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Add your virtual environment to Jupyter
python -m ipykernel install --user --name=ai-ml-env

# Select the kernel in Jupyter: Kernel → Change kernel
```

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall GPU packages
pip uninstall tensorflow torch
pip install tensorflow[and-cuda] torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```python
# Monitor memory usage
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")

# Clear variables in Jupyter
%reset

# Use smaller batch sizes
batch_size = 32  # instead of 128
```

---

## Best Practices

### Project Organization
```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── models/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

### Version Control
```bash
# Initialize Git repository
git init

# Use proper .gitignore for Python/ML
# Copy from github.com/github/gitignore/Python.gitignore

# Don't commit large files
echo "*.pkl" >> .gitignore
echo "data/" >> .gitignore
echo "models/*.h5" >> .gitignore
```

### Environment Management
```bash
# Save your environment
pip freeze > requirements.txt

# Recreate environment elsewhere
pip install -r requirements.txt

# Use virtual environments for each project
python -m venv project1-env
python -m venv project2-env
```

---

## Next Steps

Once your environment is set up:

1. **Test with a simple project**: [Getting Started Guide](../docs/getting-started.md)
2. **Try hands-on projects**: [Beginner Projects](../04-hands-on-projects/beginner/)
3. **Learn about tools**: [Tools & Technologies](../03-tools-and-technologies/)
4. **Join the community**: Contribute to discussions and help others

Remember: The perfect setup is the one you'll actually use. Start simple and improve as you learn!