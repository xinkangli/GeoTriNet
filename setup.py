"""
Setup configuration for GeoTriNet.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="geotri-net",
    version="0.1.0",
    author="GeoTriNet Contributors",
    description="Geometry-Aware Trimodal Network for Bee Toxicity Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/geotri-net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "rdkit>=2021.09.0",
        "scipy>=1.7.0",
    ],
)
