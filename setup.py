import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
files = []
setuptools.setup(
    name="phosphodisco",
    version="0.0.1",
    author="Lili Blumenberg, Ruggles Lab",
    author_email="lili.blumenberg@gmail.com",
    description="A package for phosphorylation module discovery and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liliblu/phosphodisco",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Unix",
    ],
    install_requires=[
        "pandas >= 0.24.2",
        "numpy >= 1.16.4",
        "scipy >= 1.2.1",
        "matplotlib >= 3.1.0",
        "seaborn >= 0.9.0",
        "scikit-learn >= 0.22.0",
        "statsmodels >= 0.11.0",
        "hypercluster >= 0.1.13"
    ],
    package_data={"phosphodisco": files},
    packages=setuptools.find_packages()
)
