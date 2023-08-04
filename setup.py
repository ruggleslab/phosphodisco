import setuptools

with open("README_pypi.md", "r") as fh:
    long_description = fh.read()
files = ["data/*"]
setuptools.setup(
    name="phosphodisco",
    version="0.0.1",
    author="Tobias Schraink, Ruggles Lab",
    author_email="tschraink@gmail.com",
    description="A package for phosphorylation module discovery and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruggleslab/phosphodisco",
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
        "hypercluster >= 0.1.13",
        "oyaml >= 1.0",
        "logomaker >= 0.8",
        "snakemake >= 7.8.5",
        "hdbscan >= 0.8.28",
        "cython >= 0.29.30",
    ],
    package_data={"phosphodisco": files},
    entry_points={
        "console_scripts": [
            "phdc_run=phosphodisco.cli:run",
            "phdc_generate_config=phosphodisco.cli:generate_config",
        ]
    },
    packages=setuptools.find_packages(),
)
