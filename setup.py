#!/usr/bin/python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'ndlib',
    version = '0.0.1',
    author = "Nakul",
    author_email = "_@_.abc",
    description = "DL library",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License"
        "Operating System :: OS Independent",

    ],
    python_requires = '>=3.6',
    install_requires = [
        'numpy>=1.8.0',
        'matplotlib>=3.0.0'
    ]

)