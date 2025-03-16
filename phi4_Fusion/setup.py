"""
setup.py - Package configuration for Phi-4 + PRISM fusion system

This module provides package installation configuration for the
Phi-4 + PRISM legal model fusion system.
"""

import os
from setuptools import setup, find_packages

# Read README for long description
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Phi-4 + PRISM fusion model for legal analysis"

# Read requirements
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    # Default requirements if file not found
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "networkx>=3.0",
        "plotly>=5.10.0",
        "scikit-learn>=1.2.0",
        "PyYAML>=6.0",
        "pypdf>=3.7.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.11.0",
        "tqdm>=4.64.0"
    ]

# Package data files to include
package_data = {
    'phi4_prism_fusion': [
        'config/*.yaml',
        'models/README.md',
    ],
}

setup(
    name="phi4-prism-fusion",
    version="0.1.0",
    description="Phi-4 + PRISM fusion model for legal analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Legal Tech Team",
    author_email="team@ailegaltech.com",
    url="https://github.com/ailegaltech/phi4-prism-fusion",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data=package_data,
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
            "PyMuPDF>=1.21.0",
            "Pillow>=9.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "phi4-prism-cli=phi4_prism_fusion.main:main",
        ],
    },
    # Add project URLs for more metadata
    project_urls={
        "Bug Tracker": "https://github.com/ailegaltech/phi4-prism-fusion/issues",
        "Documentation": "https://phi4-prism-fusion.readthedocs.io/",
        "Source Code": "https://github.com/ailegaltech/phi4-prism-fusion",
    },
)
