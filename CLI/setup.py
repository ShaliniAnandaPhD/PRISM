"""
PRISM - Setup script
"""

import os
import sys
from setuptools import setup, find_packages

# Define version
VERSION = '3.4.2'

# Define minimum requirements
REQUIRED = [
    'PyYAML>=6.0',
    'rich>=13.3.5',
    'click>=8.1.3',
    'numpy>=1.24.3',
    'pandas>=2.0.1',
    'tqdm>=4.65.0',
    'Jinja2>=3.1.2',
    'pydantic>=1.10.8',
]

# Define optional requirements
EXTRAS = {
    'audio': [
        'librosa>=0.10.0',
        'pydub>=0.25.1',
        'soundfile>=0.12.1',
        'praat-parselmouth>=0.4.3',
        'noisereduce>=2.0.1',
    ],
    'video': [
        'opencv-python>=4.7.0.72',
        'ffmpeg-python>=0.2.0',
        'av>=10.0.0',
        'imageio>=2.28.1',
        'scikit-image>=0.20.0',
    ],
    'document': [
        'python-docx>=0.8.11',
        'PyPDF2>=3.0.1',
        'pylatex>=1.4.1',
        'reportlab>=3.6.13',
    ],
    'ml': [
        'transformers>=4.29.2',
        'sentence-transformers>=2.2.2',
        'torch>=2.0.1',
        'torchvision>=0.15.2',
        'networkx>=3.1',
        'spacy>=3.5.3',
        'scikit-learn>=1.2.2',
    ],
    'gpu': [
        'torch-cuda>=2.0.1',
        'onnxruntime-gpu>=1.14.1',
    ],
    'full': [
        # Will include all optional dependencies
    ]
}

# Combine all extras for 'full'
EXTRAS['full'] = sorted(set(sum(EXTRAS.values(), [])))

# Read the content of README.md
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'PRISM - Protection-order Resource for Integrated Scanning and Management'

setup(
    name='prism-cli',
    version=VERSION,
    description='Protection-order Resource for Integrated Scanning and Management',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='PRISM Development Team',
    author_email='support@prism-forensics.org',
    url='https://github.com/legaltech/prism',
    packages=find_packages(exclude=['tests', 'docs']),
    entry_points={
        'console_scripts': [
            'prism=prism.prism:main',
        ],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Affero GPL v3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Legal Industry',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Office Suites',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Video :: Conversion',
    ],
    python_requires='>=3.9',
    keywords='legal, forensics, audio analysis, video analysis, evidence, protection order, domestic violence',
)
