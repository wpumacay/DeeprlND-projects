
from setuptools import find_packages
from setuptools import setup 

with open( 'requirements.txt' ) as fhandle :
    _requirements = fhandle.read().splitlines()

packages = find_packages()

setup(
    name                    = 'collaboration',
    version                 = '0.0.1',
    description             = 'Deeprl nanodegree - Project 3 - Collaboration',
    author                  = 'Wilbert Santos Pumacay Huallpa',
    license                 = 'MIT License',
    author_email            = 'wpumacay@gmail.com',
    url                     = 'https://github.com/wpumacay/DeeprlND-projects/',
    keywords                = 'rl ai dl',
    packages                = packages,
    install_requires        = _requirements
)