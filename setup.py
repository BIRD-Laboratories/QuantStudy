from setuptools import setup, find_packages

setup(
    name="qe_macro_calc",
    version="0.1",
    packages=["qe_macro_calc"],  # Explicitly specify the package directory
    install_requires=[
        "jax",
        "jaxlib",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "qe_macro_calc=qe_macro_calc.main:main",  # Adjusted entry point to reflect the package structure
        ],
    },
)