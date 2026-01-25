from setuptools import setup, find_packages

setup(
    name="universal_init",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
        "scikit-learn>=1.2.0",
    ],
)
