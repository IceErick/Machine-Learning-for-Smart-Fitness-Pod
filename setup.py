#!/usr/bin/env python3
"""
Setup script for Smart Fitness Pod Machine Learning Module
P2025-26 Project: Smart Fitness Pod with Wearable Sensor
"""

from setuptools import setup, find_packages
import pathlib

# read README.md
here = pathlib.Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

# read requirements.txt
def read_requirements():
    with open(here / "requirements-ml.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fitness-ai-ml",
    version="0.1.0",
    description="Machine Learning module for Smart Fitness Pod - Real-time exercise classification and repetition counting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Erick LI",
    author_email="erickliyuxuan@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="machine-learning, imu, exercise-classification, repetition-counting, tensorflow-lite",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <4",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fitness-train=fitness_ai.training.trainer:main",
            "fitness-convert=fitness_ai.utils.model_converter:main",
            "fitness-eval=fitness_ai.training.evaluator:main",
        ],
    },
    package_data={
        "fitness_ai": [
            "models/pretrained/*.tflite",
            "data/sample/*.csv",
        ],
    },
    include_package_data=True,
    project_urls={
        "Bug Reports": "https://github.com/IceErick/smart-fitness-pod/issues",
        "Source": "https://github.com/IceErick/smart-fitness-pod",
        "Documentation": "https://github.com/IceErick/smart-fitness-pod/docs",
    },
)