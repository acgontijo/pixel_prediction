from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if "git+" not in line]

setup(
    name="pixel_prediction",
    version="0.0.4",
    description="Pixel Prediction Model (U-Net) for deforestation detection",
#    long_description=open("README.md", encoding="utf-8").read(),
#    long_description_content_type="text/markdown",
    license="MIT",
    author="acgontijo",  # Ensure this line has no errors
    author_email="example@example.com",  # Replace or remove if not needed
    url="https://github.com/acgontijo/pixel_prediction",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
