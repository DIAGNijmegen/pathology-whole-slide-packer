from setuptools import setup, find_packages

# Read the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="wsipack",
    version="0.2.0",
    author="Witali Aswolinskiy",
    url="https://github.com/DIAGNijmegen/pathology-whole-slide-packer",
    packages=find_packages(),
    install_requires=requirements,  # Use the contents of requirements.txt
    long_description="Copy tissue sections from one or multiple whole slide images and 'pack' them together removing excessive white space.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
