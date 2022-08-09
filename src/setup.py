from setuptools import find_packages, setup
setup(
    name='mewc_common',
    packages=find_packages(include=['mewc_common']),
    version='0.1.0',
    description='MEWC Common Functions',
    author='Zach Aandahl',
    license='3-Clause BSD',
    install_requires=['pyyaml'],
)