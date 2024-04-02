from setuptools import find_packages, setup

setup(
    name='deepmaps',
    packages=find_packages(),
    py_modules=['src'],
    version='0.20',
    description='suggestor pipeline for optimal DBS settings with volumetric deep learning',
    author='Jan Waligorski',
    license='MIT',
)
