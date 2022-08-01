# library
import os
from setuptools import find_packages, setup

# variable
package_name = "rfsoc_book"

# function
def generate_pkg_dirs(pkgname):
    data_files = []
    for directory in os.walk(os.path.join(os.getcwd(), pkgname)):
        for file in directory[2]:
            data_files.append("".join([directory[0],"/",file]))
    return data_files

# setuptools
setup(
    name=package_name,
    version='1.0.0',
    install_requires=[
        "strath_sdfec @ https://github.com/strath-sdr/rfsoc_sdfec/archive/v1.0.1.tar.gz",
        "rfsoc_freqplan @ https://github.com/strath-sdr/rfsoc_frequency_planner/archive/v0.3.2.tar.gz"
    ],
    author="strath-sdr",
    packages=find_packages(),
    package_data={"" : generate_pkg_dirs(package_name)},
    description="University of Strathclyde (StrathSDR) RFSoC Book.")
