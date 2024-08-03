from setuptools import setup, find_packages

setup(
    name='qe_macro_calc',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyopencl',
    ],
    extras_require={
        'mpi': ['mpi4py'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)