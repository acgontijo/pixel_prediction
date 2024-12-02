from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='pixel_prediction',
      version="0.0.4",
      description="pixel_prediction Model (unet)",
      license="MIT",
      author="Quacso",
      author_email="",
      #url="https://github.com/acgontijo/pixel_prediction",
      install_requires=requirements,
      packages=find_packages(),

      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
