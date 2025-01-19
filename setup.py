from setuptools import setup, find_packages

setup(
    name="102203594-Topsis", 
    version="0.1.1",
    author="Shaurya Jain",
    author_email="sjain_be22@example.com",
    description="A Python implementation of the TOPSIS ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ShauryaJ123/topsis",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
