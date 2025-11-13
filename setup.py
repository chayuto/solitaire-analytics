"""Setup script for Solitaire Analytics Engine."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="solitaire-analytics",
    version="0.1.0",
    author="Solitaire Analytics Team",
    description="A Python 3.12 analytics engine for Solitaire games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chayuto/solitaire-analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment :: Board Games",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
