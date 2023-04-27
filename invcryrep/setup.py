#!/usr/bin/env python

import os
import glob

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name = 'invcryrep',
        version = '1.0.0',
        setup_requires=['setuptools>=18.0','m3gnet','scikit-learn','graphviz','pygraphviz'],
        python_requires='>=3.6',
        install_requires=["numpy>=1.14.3"],
        description=(
          "Invertible crystal representation (SLICES)"),
        author = 'Hang Xiao',
        author_email = 'xiaohang007@gmail.com',
        packages = find_packages(),
        include_package_data = True,
        platforms = 'any',
    )

