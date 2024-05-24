import os
from setuptools import setup
import json
from setuptools import setup

# load info from nixio/info.json
with open(os.path.join("etrack", "info.json")) as infofile:
    infodict = json.load(infofile)


NAME = "etrack"
VERSION = infodict["VERSION"]
AUTHOR = infodict["AUTHOR"]
CONTACT = infodict["CONTACT"]
BRIEF = infodict["BRIEF"]
HOMEPAGE = infodict["HOMEPAGE"]
CLASSIFIERS = "science"
README = "README.md"

with open(README) as f:
    description_text = f.read()
DESCRIPTION = description_text

packages = [
    "etrack", "etrack.io"
]

install_req = ["h5py", "pandas", "matplotlib", "numpy", "opencv-python"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=CONTACT,
    packages=packages,
    install_requires=install_req,
    include_package_data=True,
    long_description=description_text,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    license="BSD"
)