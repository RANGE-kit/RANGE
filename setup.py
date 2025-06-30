# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:27:07 2025

@author: d2j
"""

from setuptools import setup, find_packages

setup(
    name="RANGE_py",
    version="0.1",
    author="d2j",
    author_email="Email Here",
    description="RANGE: ",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhangdf07/RANGE",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here, e.g.
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
