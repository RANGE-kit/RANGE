# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:27:07 2025

@author: d2j
"""

from setuptools import setup, find_packages

setup(
    name="RANGE_go",
    version="1.0",
    author="Difan Zhang",
    author_email="zhangd2@ornl.gov",
    description="RANGE: a Robust Adaptive Nature-inspired Global Explorer of potential energy surfaces",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type=" ",
    url="https://github.com/RANGE-kit/RANGE",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # "numpy>=1.20",
        # "scipy>=1.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Materials Science",
        "License :: OSI Approved :: MIT License",
    ],
)
