#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst", encoding="utf-8") as history_file:
    history = history_file.read()

with open("requirements.txt", encoding="utf-8") as requirements_file:
    requirements = test_requirements = requirements_file.read().splitlines()

setup(
    author="Nikita Toloknov",
    author_email="2360402@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="ml_package_framework contains a set of classes and methods\n"
    "for simplified experimentation and packaging of models",
    entry_points={
        "console_scripts": [
            "ml_package_framework=ml_package_framework.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="ml_package_framework",
    name="ml_package_framework",
    packages=find_packages(include=["ml_package_framework", "ml_package_framework.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/amp1re/ml_package_framework",
    version="0.3.0",
    zip_safe=False,
)
