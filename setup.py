from setuptools import setup, find_packages

setup(
    name='calvin_utils',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'plotly',
        'nbformat>=4.2.0'
    ],
)
