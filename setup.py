#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical NLP Lean Package
Production-ready Medical Named Entity Recognition Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="medical-nlp-lean",
    version="1.0.0",
    description="Production-ready Medical Named Entity Recognition Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Medical NLP Team",
    author_email="medical-nlp@example.com",
    url="https://github.com/yourusername/medical-nlp-lean",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "spacy>=3.7.2",
        "negspacy>=1.0.4",
        "scispacy>=0.5.3",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "streamlit>=1.28.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-ner=scripts.run_ner:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="medical nlp ner biobert entity-recognition clinical-text",
    include_package_data=True,
    zip_safe=False,
)
