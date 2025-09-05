from setuptools import find_packages
from distutils.core import setup

setup(
    name='pcl_vae',
    version='0.0.3',
    author='Name Surname',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='name@uni',
    description='Package for training and evaluating task-driven point cloud compression VAEs',
    install_requires=['torch>=1.13',
                      'numpy']
)
