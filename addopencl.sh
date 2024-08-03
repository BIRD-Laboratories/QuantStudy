#!/bin/bash

# Update and install necessary packages
echo "Updating package list and installing necessary packages..."
apt-get update
apt-get install -y cmake build-essential ocl-icd-opencl-dev clinfo pocl-opencl-icd

# Set environment variables
echo "Setting environment variables..."
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Verify OpenCL installation
echo "Verifying OpenCL installation..."
clinfo

echo "PoCL OpenCL support for CPU has been added to your Google Colab environment."