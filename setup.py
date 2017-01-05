from setuptools import setup

with open("requirements.txt", "r") as f:
    dependencies = f.read().split()

setup(
    name="mlfromscratch",
    version="0.1",
    packages=["mlfromscratch",],
    author="Jonathan Arfa",
    license="Apache-2.0",
    install_requires=dependencies,
)
