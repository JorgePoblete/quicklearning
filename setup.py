
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quicklearning",
    version="0.0.19",
    license="MIT",
    author="koke",
    author_email="jpobleteriquelme@gmail.com",
    description="Create Tensorflow models and train them fast and easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JorgePoblete/quicklearning",
    py_modules=["quicklearning"],
    package_dir={"":"src"},
    packages=[
        "quicklearning.classification.image"
    ],
    install_requires = [
        "matplotlib >= 3.3.2",
        "Pillow >= 7.2.0",
        "DuckDuckGoImages >= 2.0.0",
        "tensorflow >= 2.3.0",
        "tensorflow_hub >= 0.9.0"
    ],
    extras_require = {
        "dev": [
            "pytest >= 3.7"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 
