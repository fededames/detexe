from pathlib import Path
import os
from setuptools import find_packages, setup

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="detexe",
    version="0.0.2.2",
    description="A framework to create malware detectors based on machine learning.",
    keywords=["malware, pe, machine learning, static analysis, adversarial attack"],
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Federico Damian",
    author_email="fededames@gmail.com",
    url="https://github.com/fededames/detexe",
    license="GPL 3.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=(
        "capstone >= 4.0.2",
        "deap >= 1.3.1",
        "gensim >= 4.1.2",
        "lief >= 0.11.0",
        "lightgbm >= 3.3.0",
        "matplotlib >= 3.3.4",
        "nltk >= 3.6.3",
        "numpy >= 1.19.5",
        "pandas >= 1.1.5",
        "python_magic >= 0.4.24",
        "scikit_learn >= 1.0.1",
        "scikit_optimize >= 0.9.0",
        "secml >= 0.14",
        "setuptools >= 57.0.0",
    ),
    extras_require={"dev": ["black", "flake8", "isort", "wheel"]},
    entry_points={"console_scripts": ["detexe=detexe.cli:parse_args"]},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
        'Topic :: Security :: Cryptography',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
    ]
)
