#!/usr/bin/env python

import os,subprocess
import glob
import codecs
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
module_dir = os.path.dirname(os.path.abspath(__file__))
# these things are needed for the README.md show on pypi
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '1.4.4'
DESCRIPTION = "Invertible crystal representation (SLICES)"





if __name__ == "__main__":
    setup(
        name = 'slices',
        version = VERSION,
        setup_requires=[],
        python_requires='>=3.8',
        install_requires=['setuptools>=18.0','numpy==1.23.5','m3gnet==0.2.4','scikit-learn<=1.3.1', 'smact'],
        description=DESCRIPTION,
        long_description_content_type="text/markdown",
        long_description=long_description,
        author = 'Hang Xiao',
        author_email = 'xiaohang07@live.cn',
        packages = find_packages(),

        package_data={"invcryrep": ["nlopt.h", "xtb_noring_nooutput_nostdout_noCN","./MP-2021.2.8-EFS/checkpoint",\
        "./MP-2021.2.8-EFS/m3gnet.data-00000-of-00001","./MP-2021.2.8-EFS/m3gnet.index","./MP-2021.2.8-EFS/m3gnet.json"]},
        platforms = 'any',
        license="GPL 2.1",
    )

