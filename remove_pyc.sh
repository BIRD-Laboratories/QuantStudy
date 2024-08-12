#!/bin/bash

# Function to remove .pyc files and empty directories
remove_pyc_files() {
    find . -type f -name '*.pyc' -exec rm -f {} +
    find . -type d -name '__pycache__' -exec rmdir {} + 2>/dev/null
}

# Function to remove *.egg.info directories and /build directory
remove_additional_files() {
    find . -type d -name '*.egg-info' -exec rm -rf {} +
    rm -rf ./build
}

# Call the functions
remove_pyc_files
remove_additional_files

echo "All .pyc files, empty __pycache__ directories, *.egg-info directories, and /build directory have been removed."