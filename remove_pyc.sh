#!/bin/bash

# Function to remove .pyc files and empty directories
remove_pyc_files() {
    find . -type f -name '*.pyc' -exec rm -f {} +
    find . -type d -name '__pycache__' -exec rmdir {} + 2>/dev/null
}

# Call the function
remove_pyc_files

echo "All .pyc files and empty __pycache__ directories have been removed."