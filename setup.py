from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yolo-cell-count",
    version="1.0.0",
    author="Juan Jose Cas",
    description="Real-time cell detection and counting using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juanjosecas/YOLO-cell-count",
    py_modules=["live_script", "LiveApp"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.0.0",
        "ultralytics>=8.0.0",
        "numpy>=1.19.0",
        "psutil>=5.0.0",
    ],
    extras_require={
        "gui": [
            "tk>=0.1.0",
        ],
    },
)
